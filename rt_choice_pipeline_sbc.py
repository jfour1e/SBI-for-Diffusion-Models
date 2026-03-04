"""Simulation-Based Calibration (SBC) for a pretrained NPE model."""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn

from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS
MODEL_PATH = os.path.expanduser("~/models/npe_rt_choice.pt")

PARAM_NAMES = ["a0", "lam", "v", "B", "tau"]


def load_npe(model_path: str, device: str = "cpu"):
    """Rebuild the exact NPE architecture and load saved weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P + 1
    theta_dim = 5
    x_dim = T * trial_dim

    embedding_net = PermutationInvariantEmbedding(
        num_trials=T,
        trial_dim=trial_dim,
        trial_net_hidden=int(saved_cfg.NPE_TRIAL_NET_HIDDEN),
        trial_net_layers=int(saved_cfg.NPE_TRIAL_NET_LAYERS),
        trial_net_output_dim=int(saved_cfg.NPE_TRIAL_NET_OUTPUT_DIM),
        post_agg_hidden=int(saved_cfg.NPE_POST_AGG_HIDDEN),
        post_agg_layers=int(saved_cfg.NPE_POST_AGG_LAYERS),
        output_dim=int(saved_cfg.NPE_EMBEDDING_OUTPUT_DIM),
        aggregation=str(saved_cfg.NPE_AGG_FN),
    )

    est_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=int(saved_cfg.NPE_HIDDEN_FEATURES),
        num_transforms=int(saved_cfg.NPE_NUM_TRANSFORMS),
        num_bins=int(saved_cfg.NPE_NUM_BINS),
        embedding_net=embedding_net,
    )

    # Instantiate the nn.Module with dummy data (required by sbi's builder)
    dummy_theta = torch.randn(2, theta_dim, device=device)
    dummy_x = torch.randn(2, x_dim, device=device)
    density_estimator = est_builder(dummy_theta, dummy_x)

    density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    density_estimator.to(device)
    density_estimator.eval()

    return density_estimator, saved_cfg


@torch.no_grad()
def run_sbc(
    posterior: DirectPosterior,
    prior_theta,
    saved_cfg,
    *,
    R: int = 200,          # number of SBC replications (bigger = better)
    N: int = 500,          # posterior samples per replication
    seed: int = 0,
    device: str = "cpu",
    show_progress: bool = True,
):
    """
    Returns:
      ranks: np.ndarray of shape (R, theta_dim), values in {0,1,...,N}
             rank = number of posterior samples < true theta for each dim.
    """
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)

    ranks = np.zeros((R, 5), dtype=np.int32)

    for r in range(R):
        if show_progress and (r % max(1, R // 10) == 0):
            print(f"SBC {r}/{R}...")

        # Sample ground-truth theta from the prior
        theta_true = prior_theta.sample((1,)).view(1, 5).to(dev, dtype=torch.float32)

        # Simulate one observed session x ~ p(x | theta_true)
        _, x_o_flat = simulate_training_sessions(
            prior_theta=prior_theta,
            num_sessions=1,
            num_trials=T,
            device=dev,
            mu_sensory=float(saved_cfg.MU_SENSORY),
            p_success=float(saved_cfg.P_SUCCESS),
            P=P,
            log_rt=bool(saved_cfg.LOG_RT_MANUALLY),
            seed=int(rng.integers(0, 2**31 - 1)),
            theta=theta_true,
            sim_batch_size=1,
        )
        x_o_flat = x_o_flat.to(dev, dtype=torch.float32)

        # Sample posterior draws theta^(i) ~ p(theta | x_o)
        post_samps = posterior.sample((N,), x=x_o_flat, show_progress_bars=False)  # (N,5)
        post_samps = post_samps.detach().cpu().numpy()
        theta_true_np = theta_true.detach().cpu().numpy().reshape(-1)  # (5,)

        # Rank statistic per dim: count posterior samples strictly less than theta_true
        # Produces integer in [0, N]
        ranks[r, :] = (post_samps < theta_true_np[None, :]).sum(axis=0).astype(np.int32)

    return ranks


def plot_sbc_ranks(ranks: np.ndarray, *, N: int, outdir: str = "sbc_outputs"):
    os.makedirs(outdir, exist_ok=True)

    theta_dim = ranks.shape[1]
    bins = np.arange(N + 2) - 0.5  # bins centered on integers 0..N

    fig, axes = plt.subplots(1, theta_dim, figsize=(3.2 * theta_dim, 3.0), sharey=True)
    if theta_dim == 1:
        axes = [axes]

    for j in range(theta_dim):
        ax = axes[j]
        ax.hist(ranks[:, j], bins=bins, density=True)
        ax.set_title(PARAM_NAMES[j])
        ax.set_xlabel("rank")
        ax.set_xlim(-0.5, N + 0.5)
        if j == 0:
            ax.set_ylabel("density")

        # Reference uniform density line (approx)
        ax.axhline(1.0 / (N + 1), linewidth=2)

    plt.tight_layout()
    fig_path = os.path.join(outdir, "sbc_rank_histograms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"Loading model from {MODEL_PATH} ...")
    density_estimator, saved_cfg = load_npe(MODEL_PATH, device=device)
    print("Model loaded.")

    # Prior
    prior_theta = build_prior_theta()
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)  # do not reassign

    # Direct posterior wrapper
    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    # SBC knobs
    R = 1000   # replications; start 100–200
    N = 500   # posterior samples per replication; 200–1000 is common

    print(f"Running SBC with R={R}, N={N} on device={device} ...")
    ranks = run_sbc(
        posterior=posterior,
        prior_theta=prior_theta,
        saved_cfg=saved_cfg,
        R=R,
        N=N,
        seed=123,
        device=device,
        show_progress=True,
    )

    outdir = os.environ.get("OUTDIR", "sbc_outputs")
    npy_path = os.path.join(outdir, "sbc_ranks.npy")
    os.makedirs(outdir, exist_ok=True)
    np.save(npy_path, ranks)
    print("Saved:", npy_path)

    plot_sbc_ranks(ranks, N=N, outdir=outdir)


if __name__ == "__main__":
    main()