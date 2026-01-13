from __future__ import annotations

import os 
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt 
import torch

from sbi.inference import MNLE, MCMCPosterior
from sbi.neural_nets import likelihood_nn
from sbi.utils import mcmc_transform

from sbi_for_diffusion_models.potentials import ThetaOnlyPosteriorPotential, ConditionedMNLELogLikelihood
from sbi_for_diffusion_models.models.rt_choice_model import simulate_session_data_rt_choice
from sbi_for_diffusion_models.mnle import run_inference_mcmc

def train_mnle(cfg, proposal_z, z_train, x_train, device: str = "cpu"):
    """
    Train an MNLE density estimator on simulations (z_train, x_train).

    Args:
        cfg: RunConfig (or any object with the required attributes)
        proposal_z: distribution over z = [theta, pulses]
        z_train: (N, 5+P) tensor
        x_train: (N, 2) tensor
        device: "cpu" or "cuda"

    Returns:
        density_estimator: trained conditional density estimator
    """

    est_builder = likelihood_nn(
        model="mnle",
        log_transform_x=bool(cfg.SBI_LOG_TRANSFORM_X),
        z_score_theta="independent",
        z_score_x=cfg.Z_SCORE_X,
        hidden_features=128,
        num_transforms=10,
        num_bins=24,
    )

    trainer = MNLE(prior=proposal_z, density_estimator=est_builder, device=device)
    trainer = trainer.append_simulations(z_train, x_train)

    # sbi changed arg name across versions
    try:
        density_estimator = trainer.train(training_batch_size=cfg.TRAIN_BATCH_SIZE)
    except TypeError:
        density_estimator = trainer.train(batch_size=cfg.TRAIN_BATCH_SIZE)

    return density_estimator

def run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o) -> torch.Tensor:
    """
    Runs MCMC over global theta (dim=5) conditioned on trial-wise pulses_o,
    using an MNLE density_estimator trained on z=[theta,pulses].

    Returns:
        samples: (cfg.POSTERIOR_SAMPLES, 5) on CPU
    """

    conditioned_loglike = ConditionedMNLELogLikelihood(
        estimator=density_estimator,
        local_theta=pulses_o,
        device="cpu",
    )

    potential_theta = ThetaOnlyPosteriorPotential(
        conditioned_loglike=conditioned_loglike,
        prior_theta=prior_theta,
        x_o=x_o,
        device="cpu",
        temperature=cfg.TEMPERATURE,
    )

    theta_transform = mcmc_transform(prior_theta)

    posterior = MCMCPosterior(
        potential_fn=potential_theta,
        proposal=prior_theta,
        theta_transform=theta_transform,
        method="nuts_pyro",
        num_chains=cfg.NUM_CHAINS,
        warmup_steps=cfg.WARMUP_STEPS,
        thin=1,
        init_strategy="proposal",
        num_workers=1,
    )

    samples = posterior.sample(
        (cfg.POSTERIOR_SAMPLES,),
        x=x_o,
        show_progress_bars=True,
    ).detach().cpu()

    return samples


def _compute_ranks(theta_true: torch.Tensor, posterior_samples: torch.Tensor) -> torch.Tensor:
    """
    SBC rank for each dimension:
      rank_d = #{s in posterior_samples[:, d] : s < theta_true[d]}
    """
    theta_true = theta_true.view(-1)
    return (posterior_samples < theta_true[None, :]).sum(dim=0).to(torch.int64)


def _plot_sbc_rank_histograms(
    ranks: np.ndarray,  # (num_datasets, D)
    *,
    param_names: Sequence[str],
    outpath: Optional[str] = None,
    bins: int = 30,
):
    D = ranks.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.5 * D), constrained_layout=True)
    if D == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.hist(ranks[:, d], bins=bins)
        ax.set_title(f"SBC ranks: {param_names[d]}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print("Saved SBC plot:", outpath)

    return fig

def run_sbc(
    cfg,
    *,
    prior_theta,
    density_estimator,
    device: str = "cpu",
    num_datasets: int = 25,
    posterior_samples_per_dataset: Optional[int] = None,
    seed: int = 0,
    param_names: Sequence[str] = ("a0", "lam", "v", "B", "tau"),
    outdir: str = "sbc_outputs",
    plot_bins: int = 30,
) -> dict:
    """
    Simulation-Based Calibration (SBC) for your MNLE + MCMC pipeline.

    For each dataset:
      1) sample theta_true ~ prior
      2) simulate (x_o, pulses_o) using simulate_session_data_rt_choice(..., return_pulse_sides=True)
      3) run MCMC posterior samples using your run_inference_mcmc
      4) compute SBC ranks per parameter dimension

    Returns dict with:
      - thetas_true: (N, 5)
      - ranks: (N, 5)
      - all_samples: list of length N, each tensor (S,5)
    """
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # If you want to override the number of posterior samples for SBC without mutating cfg,
    # we’ll just pass a temporary cfg-like shim when needed.
    cfg_for_inference = cfg
    if posterior_samples_per_dataset is not None:
        # Make a lightweight wrapper that reads everything from cfg except POSTERIOR_SAMPLES
        class _CfgShim:
            def __init__(self, base, S):
                self._base = base
                self.POSTERIOR_SAMPLES = int(S)

            def __getattr__(self, name):
                return getattr(self._base, name)

        cfg_for_inference = _CfgShim(cfg, posterior_samples_per_dataset)

    thetas_true = []
    ranks = []
    all_samples = []

    for i in range(num_datasets):
        # 1) theta_true ~ prior
        theta_true = prior_theta.sample((1,)).view(5).to(torch.float32).cpu()

        # 2) simulate observed dataset (x_o, pulses_o) using your existing function
        # Important: pass a dataset-specific rng to ensure pulses differ each iteration
        ds_seed = int(rng.integers(0, 2**31 - 1))
        ds_rng = np.random.default_rng(ds_seed)

        x_raw, pulses_o = simulate_session_data_rt_choice(
            theta_true,
            int(cfg.NUM_TRIALS_OBS),
            rng=ds_rng,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
            return_pulse_sides=True,
        )
        # x_raw is (T,2) [rt, choice]; your inference code expects packed x if you used packing in training
        # In your current pipeline: training uses pack_x_rt_choice after sim, and observed uses pack_x_rt_choice too.
        # simulate_session_data_rt_choice returns already [rt, choice] in same format (float32; choice 0/1/2),
        # which matches your pack format when LOG_RT_MANUALLY=False. If LOG_RT_MANUALLY=True, you'd need to log rt here.
        x_o = x_raw.detach().cpu().to(torch.float32)

        if bool(cfg.LOG_RT_MANUALLY):
            x_o = x_o.clone()
            x_o[:, 0] = torch.log(x_o[:, 0].clamp_min(1e-6))

        pulses_o = pulses_o.detach().cpu().to(torch.float32)

        # 3) infer posterior samples
        samples = run_inference_mcmc(cfg_for_inference, prior_theta, density_estimator, x_o, pulses_o)  # (S,5)

        # 4) ranks
        r = _compute_ranks(theta_true, samples)

        thetas_true.append(theta_true.numpy())
        ranks.append(r.numpy())
        all_samples.append(samples)

        print(f"[SBC] {i+1:>3}/{num_datasets} done. ranks={r.tolist()}")

    thetas_true = np.stack(thetas_true, axis=0)
    ranks = np.stack(ranks, axis=0)

    # Save
    np.save(os.path.join(outdir, "sbc_thetas_true.npy"), thetas_true)
    np.save(os.path.join(outdir, "sbc_ranks.npy"), ranks)
    print("Saved:", os.path.join(outdir, "sbc_thetas_true.npy"))
    print("Saved:", os.path.join(outdir, "sbc_ranks.npy"))

    # Plot
    _plot_sbc_rank_histograms(
        ranks,
        param_names=param_names,
        outpath=os.path.join(outdir, "sbc_rank_histograms.png"),
        bins=plot_bins,
    )

    return {"thetas_true": thetas_true, "ranks": ranks, "all_samples": all_samples}