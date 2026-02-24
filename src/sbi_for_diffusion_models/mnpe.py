from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses, pack_x_rt_choice
from sbi_for_diffusion_models.data_simulator import flatten_observed_session, simulate_training_sessions
from sbi_for_diffusion_models.mnle import _compute_ranks, _plot_sbc_rank_histograms
from sbi_for_diffusion_models.Embeddings import MaskAwarePermutationInvariantEmbedding


# ── Training ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def simulate_npe_training_data(
    cfg,
    prior_theta,
    *,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper: simulate (theta_train, x_train) for MNPE on the chosen device.

    Returns:
      theta_train: (N_sessions, 5) on device
      x_train: (N_sessions, T*(2+P+1)) on device, with mask included per trial
    """
    P = max_num_pulses()
    theta_train, x_train = simulate_training_sessions(
        prior_theta,
        num_sessions=int(cfg.NPE_NUM_TRAIN_SESSIONS),
        num_trials=int(cfg.NUM_TRIALS_OBS),
        device=device,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(seed),
    )
    return theta_train, x_train

def train_npe_session(
    cfg,
    prior_theta,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    device: str = "cpu",
):
    """
    Train an NPE density estimator with a mask-aware permutation-invariant embedding.
    """
    dev = torch.device(device)

    # Ensure training tensors are on the correct device 
    theta_train = theta_train.to(device=dev, dtype=torch.float32)
    x_train = x_train.to(device=dev, dtype=torch.float32)

    P = max_num_pulses()
    trial_dim = 2 + P + 1
    if x_train.shape[1] % trial_dim != 0:
        raise ValueError(
            f"x_train second dim must be divisible by trial_dim={trial_dim}, "
            f"got x_train.shape={tuple(x_train.shape)}"
        )
    num_trials = x_train.shape[1] // trial_dim

    embedding_net = MaskAwarePermutationInvariantEmbedding(
        num_trials=int(num_trials),
        trial_dim=int(trial_dim),
        trial_net_hidden=int(cfg.NPE_TRIAL_NET_HIDDEN),
        trial_net_layers=int(cfg.NPE_TRIAL_NET_LAYERS),
        trial_net_output_dim=int(cfg.NPE_TRIAL_NET_OUTPUT_DIM),
        post_agg_hidden=int(cfg.NPE_POST_AGG_HIDDEN),
        post_agg_layers=int(cfg.NPE_POST_AGG_LAYERS),
        output_dim=int(cfg.NPE_EMBEDDING_OUTPUT_DIM),
        aggregation=str(cfg.NPE_AGG_FN),
    ).to(dev)

    est_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=int(cfg.NPE_HIDDEN_FEATURES),
        num_transforms=int(cfg.NPE_NUM_TRANSFORMS),
        num_bins=int(cfg.NPE_NUM_BINS),
        embedding_net=embedding_net,
    )

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    inference = NPE(prior=prior_theta, density_estimator=est_builder, device=str(dev))
    inference = inference.append_simulations(theta_train, x_train)

    try:
        density_estimator = inference.train(training_batch_size=int(cfg.NPE_TRAIN_BATCH_SIZE))
    except TypeError:
        density_estimator = inference.train(batch_size=int(cfg.NPE_TRAIN_BATCH_SIZE))

    return density_estimator, inference

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference_npe(cfg, inference_obj, density_estimator, x_o_flat, prior_theta):
    """
    Direct posterior sampling from the amortized NPE posterior.

    x_o_flat must be shaped (1, T*(2+P+1)) and include mask.
    """
    posterior = inference_obj.build_posterior(
        density_estimator=density_estimator,
        prior=prior_theta,
    )

    dev = next(density_estimator.parameters()).device
    x_o_flat = x_o_flat.to(dev, dtype=torch.float32)

    samples = (
        posterior.sample(
            (int(cfg.NPE_POSTERIOR_SAMPLES),),
            x=x_o_flat,
            show_progress_bars=True,
        )
        .detach()
        .cpu()
    )
    return samples


# ── SBC ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_sbc_npe(
    cfg,
    *,
    prior_theta,
    posterior,
    device: str = "cpu",
    num_datasets: int = 100,
    posterior_samples_per_dataset: Optional[int] = None,
    seed: int = 0,
    param_names: Sequence[str] = ("a0", "lam", "v", "B", "tau"),
    outdir: str = "sbc_npe_outputs",
    plot_bins: int = 30,
) -> dict:
    """
    SBC for the NPE pipeline using the same MNPE session simulator (mask-aware, timeout-dropped).

    This version keeps simulation and posterior sampling on the chosen device and only moves
    results to CPU for saving/plotting.
    """
    os.makedirs(outdir, exist_ok=True)

    dev = torch.device(device)
    S = int(posterior_samples_per_dataset or cfg.NPE_SBC_POST_SAMPLES)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    thetas_true = []
    ranks = []
    all_samples = []

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P + 1

    for i in range(int(num_datasets)):
        # Sample theta_true
        theta_true = prior_theta.sample((1,)).view(5).to(device=dev, dtype=torch.float32)

        # simulate data 
        _, x_o = simulate_training_sessions(
            prior_theta,
            num_sessions=1,
            num_trials=T,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
            P=P,
            log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=int(rng.integers(0, 2**31 - 1)),
            theta=theta_true,  
        )
        
        # sample from posterior 
        if x_o.shape != (1, T * trial_dim):
            raise RuntimeError(f"Unexpected x_o shape: {tuple(x_o.shape)}")

        samples = posterior.sample((S,), x=x_o, show_progress_bars=False).detach().cpu()
        r = _compute_ranks(theta_true.detach().cpu().view(5), samples)

        thetas_true.append(theta_true.detach().cpu().numpy())
        ranks.append(r.numpy())
        all_samples.append(samples)

        if (i + 1) % 10 == 0:
            print(f"[SBC-NPE] {i + 1}/{num_datasets} done. ranks={r.tolist()}")
            
    thetas_true = np.stack(thetas_true, axis=0)
    ranks = np.stack(ranks, axis=0)

    np.save(os.path.join(outdir, "sbc_thetas_true.npy"), thetas_true)
    np.save(os.path.join(outdir, "sbc_ranks.npy"), ranks)
    print("Saved:", os.path.join(outdir, "sbc_thetas_true.npy"))
    print("Saved:", os.path.join(outdir, "sbc_ranks.npy"))

    _plot_sbc_rank_histograms(
        ranks,
        param_names=param_names,
        outpath=os.path.join(outdir, "sbc_rank_histograms.png"),
        bins=int(plot_bins),
    )

    return {"thetas_true": thetas_true, "ranks": ranks, "all_samples": all_samples}