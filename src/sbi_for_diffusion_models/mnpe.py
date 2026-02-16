from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding
from sbi.neural_nets.embedding_nets.permutation_invariant import (
    PermutationInvariantEmbedding,
)

from sbi_for_diffusion_models.models.rt_choice_model import (
    max_num_pulses,
    simulate_session_data_rt_choice,
    pack_x_rt_choice,
)
from sbi_for_diffusion_models.data_simulator import flatten_observed_session
from sbi_for_diffusion_models.mnle import _compute_ranks, _plot_sbc_rank_histograms


# ── Permutation-invariant embedding for session data ──────────────────────────


class SessionEmbeddingNet(nn.Module):
    """Reshape flat session vector to 3-D and apply a PermutationInvariantEmbedding.

    sbi's ``posterior_nn`` feeds flat (batch, x_dim) tensors to the embedding
    network, but ``PermutationInvariantEmbedding`` expects (batch, T, trial_dim).
    This wrapper bridges the two.
    """

    def __init__(
        self,
        num_trials: int,
        trial_dim: int,
        trial_net_hidden: int = 128,
        trial_net_layers: int = 3,
        trial_net_output_dim: int = 64,
        aggregation_fn: str = "mean",
        post_agg_hidden: int = 128,
        post_agg_layers: int = 2,
        output_dim: int = 64,
    ):
        super().__init__()
        self.num_trials = num_trials
        self.trial_dim = trial_dim

        trial_net = FCEmbedding(
            input_dim=trial_dim,
            output_dim=trial_net_output_dim,
            num_layers=trial_net_layers,
            num_hiddens=trial_net_hidden,
        )

        self.perm_inv_net = PermutationInvariantEmbedding(
            trial_net=trial_net,
            trial_net_output_dim=trial_net_output_dim,
            aggregation_fn=aggregation_fn,
            num_hiddens=post_agg_hidden,
            num_layers=post_agg_layers,
            output_dim=output_dim,
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        batch_size = x_flat.shape[0]
        x_3d = x_flat.view(batch_size, self.num_trials, self.trial_dim)
        return self.perm_inv_net(x_3d)


# ── Training ──────────────────────────────────────────────────────────────────


def train_npe_session(cfg, prior_theta, theta_train, x_train, device: str = "cpu"):
    """Train an NPE density estimator with a permutation-invariant embedding.

    Args:
        cfg: RunConfig
        prior_theta: prior over theta (5 params)
        theta_train: (N, 5)
        x_train: (N, T*(2+P)) flattened session data
        device: ``"cpu"`` or ``"cuda"``

    Returns:
        (density_estimator, inference_obj)
    """
    P = max_num_pulses()
    trial_dim = 2 + P
    num_trials = x_train.shape[1] // trial_dim

    embedding_net = SessionEmbeddingNet(
        num_trials=num_trials,
        trial_dim=trial_dim,
        trial_net_hidden=cfg.NPE_TRIAL_NET_HIDDEN,
        trial_net_layers=cfg.NPE_TRIAL_NET_LAYERS,
        trial_net_output_dim=cfg.NPE_TRIAL_NET_OUTPUT_DIM,
        aggregation_fn=cfg.NPE_AGG_FN,
        post_agg_hidden=cfg.NPE_POST_AGG_HIDDEN,
        post_agg_layers=cfg.NPE_POST_AGG_LAYERS,
        output_dim=cfg.NPE_EMBEDDING_OUTPUT_DIM,
    )

    est_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=cfg.NPE_HIDDEN_FEATURES,
        num_transforms=cfg.NPE_NUM_TRANSFORMS,
        num_bins=cfg.NPE_NUM_BINS,
        embedding_net=embedding_net,
    )

    # sbi requires the prior to live on the training device
    if device != "cpu" and hasattr(prior_theta, "to"):
        prior_theta.to(device)

    inference = NPE(prior=prior_theta, density_estimator=est_builder, device=device)
    inference = inference.append_simulations(theta_train, x_train)

    try:
        density_estimator = inference.train(
            training_batch_size=cfg.NPE_TRAIN_BATCH_SIZE
        )
    except TypeError:
        density_estimator = inference.train(
            batch_size=cfg.NPE_TRAIN_BATCH_SIZE
        )

    return density_estimator, inference


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference_npe(cfg, inference_obj, density_estimator, x_o_flat, prior_theta):
    """Direct posterior sampling from the amortized NPE posterior.

    Args:
        cfg: RunConfig
        inference_obj: the NPE inference object (from ``train_npe_session``)
        density_estimator: trained estimator
        x_o_flat: (1, T*(2+P)) flattened observed session
        prior_theta: prior distribution

    Returns:
        samples: (cfg.NPE_POSTERIOR_SAMPLES, 5) on CPU
    """
    posterior = inference_obj.build_posterior(
        density_estimator=density_estimator,
        prior=prior_theta,
    )

    # ensure observation lives on the same device as the model
    device = next(density_estimator.parameters()).device
    x_o_flat = x_o_flat.to(device)

    samples = posterior.sample(
        (cfg.NPE_POSTERIOR_SAMPLES,),
        x=x_o_flat,
        show_progress_bars=True,
    ).detach().cpu()

    return samples


# ── SBC ───────────────────────────────────────────────────────────────────────


def run_sbc_npe(
    cfg,
    *,
    prior_theta,
    inference_obj,
    density_estimator,
    device: str = "cpu",
    num_datasets: int = 100,
    posterior_samples_per_dataset: Optional[int] = None,
    seed: int = 0,
    param_names: Sequence[str] = ("a0", "lam", "v", "B", "tau"),
    outdir: str = "sbc_npe_outputs",
    plot_bins: int = 30,
) -> dict:
    """SBC for the NPE pipeline (no MCMC — fast)."""
    os.makedirs(outdir, exist_ok=True)

    posterior = inference_obj.build_posterior(
        density_estimator=density_estimator,
        prior=prior_theta,
    )

    S = posterior_samples_per_dataset or cfg.NPE_SBC_POST_SAMPLES
    P = max_num_pulses()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    thetas_true = []
    ranks = []
    all_samples = []

    for i in range(num_datasets):
        theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)

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
        x_packed = pack_x_rt_choice(x_raw, log_rt=bool(cfg.LOG_RT_MANUALLY))
        x_o_flat = flatten_observed_session(x_packed, pulses_o)
        x_o_flat = x_o_flat.to(next(density_estimator.parameters()).device)

        samples = posterior.sample(
            (S,), x=x_o_flat, show_progress_bars=False,
        ).detach().cpu()

        r = _compute_ranks(theta_true, samples)
        thetas_true.append(theta_true.numpy())
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
        bins=plot_bins,
    )

    return {"thetas_true": thetas_true, "ranks": ranks, "all_samples": all_samples}
