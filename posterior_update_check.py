"""
Posterior Update Check.

For a pretrained NPE, test whether the posterior actually moves relative to
the prior on UNSEEN simulated data (data not seen during training).

For each of N_TEST datasets:
  1. Draw theta_true ~ prior
  2. Simulate a session from theta_true
  3. Sample the posterior p(theta | x)
  4. Compute per-parameter posterior contraction:
       contraction_d = 1 - std(posterior_d) / std(prior_d)
     A contraction near 1 = posterior is tight; near 0 = posterior ~= prior.

Plots:
  1. Contraction per parameter (boxplot across test datasets)
  2. Overlay: prior marginals vs posterior marginals (one representative session)
  3. Pairplot of posterior vs theta_true for a few sessions

Usage
-----
  python posterior_update_check.py
  MODEL_NAME=base python posterior_update_check.py
  N_TEST=50 python posterior_update_check.py
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn

from sbi_for_diffusion_models.priors import build_prior_theta, build_prior_theta_lapse
from sbi_for_diffusion_models.models.rt_choice_model import simulate_rt_choice_batch, max_num_pulses
from sbi_for_diffusion_models.models.lapse_rt_choice_model import simulate_rt_choice_batch_lapse
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

MODEL_NAME  = os.environ.get("MODEL_NAME",  "lapse")
N_TEST      = int(os.environ.get("N_TEST",  "20"))
N_POST      = int(os.environ.get("N_POST",  "2000"))
N_PRIOR     = int(os.environ.get("N_PRIOR", "5000"))
SEED        = int(os.environ.get("SEED",    "99"))
OUTDIR      = os.environ.get("OUTDIR",      "posterior_update_outputs")
MODEL_DIR   = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models"))


def get_spec(model_name: str) -> dict:
    if model_name == "base":
        return dict(
            prior_builder=build_prior_theta,
            simulate_batch_fn=simulate_rt_choice_batch,
            param_names=["a0", "lam", "v", "B", "tau"],
            model_file="npe_rt_choice_base.pt",
        )
    elif model_name == "lapse":
        return dict(
            prior_builder=build_prior_theta_lapse,
            simulate_batch_fn=simulate_rt_choice_batch_lapse,
            param_names=["a0", "lam", "v", "B", "tau", "p_lapse"],
            model_file="npe_rt_choice_lapse.pt",
        )
    raise ValueError(model_name)


def load_npe(model_path: str, prior_theta, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg  = checkpoint["config"]
    dev        = torch.device(device)

    P          = max_num_pulses()
    T          = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim  = 2 + P
    theta_dim  = int(torch.as_tensor(prior_theta.sample((1,))).reshape(-1).numel())

    emb = PermutationInvariantEmbedding(
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
        embedding_net=emb,
    )
    x_dim = T * trial_dim
    de = est_builder(
        torch.randn(2, theta_dim, device=dev),
        torch.randn(2, x_dim,    device=dev),
    )
    de.load_state_dict(checkpoint["state_dict"], strict=True)
    de.to(dev).eval()
    return de, saved_cfg


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev    = torch.device(device)
    print(f"Model: {MODEL_NAME}  |  N_test: {N_TEST}  |  device: {device}")

    spec              = get_spec(MODEL_NAME)
    prior_theta       = spec["prior_builder"]()
    simulate_batch_fn = spec["simulate_batch_fn"]
    param_names       = spec["param_names"]
    D                 = len(param_names)

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    model_path = os.path.join(MODEL_DIR, spec["model_file"])
    print(f"Loading model from {model_path} ...")
    de, saved_cfg = load_npe(model_path, prior_theta, device)

    posterior = DirectPosterior(
        posterior_estimator=de,
        prior=prior_theta,
        device=device,
    )

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    os.makedirs(OUTDIR, exist_ok=True)

    # --- Prior samples for reference std ---
    prior_samples = torch.as_tensor(
        prior_theta.sample((N_PRIOR,)), dtype=torch.float32
    ).cpu()
    prior_std = prior_samples.std(dim=0).numpy()  # (D,)

    # --- Per-test-dataset posterior samples ---
    all_contractions = np.zeros((N_TEST, D))
    all_post_stds    = np.zeros((N_TEST, D))
    all_theta_true   = np.zeros((N_TEST, D))
    all_post_samples = []  # keep first 5 for plotting

    print(f"\nRunning {N_TEST} posterior update checks ...")
    for i in range(N_TEST):
        theta_true = torch.as_tensor(
            prior_theta.sample((1,)), device=dev, dtype=torch.float32
        ).reshape(-1)

        _, x_flat = simulate_training_sessions(
            prior_theta=prior_theta,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=simulate_batch_fn,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
            P=P,
            log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=SEED + i,
            theta=theta_true,
        )

        x_flat = x_flat.to(device, dtype=torch.float32)
        post_samples = posterior.sample(
            (N_POST,), x=x_flat, show_progress_bars=False
        ).detach().cpu()

        post_std = post_samples.std(dim=0).numpy()
        contraction = 1.0 - post_std / (prior_std + 1e-12)

        all_contractions[i] = contraction
        all_post_stds[i]    = post_std
        all_theta_true[i]   = theta_true.cpu().numpy()

        if i < 5:
            all_post_samples.append(post_samples.numpy())

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_TEST}]  mean contraction: "
                  f"{contraction.mean():.3f}")

    fig, ax = plt.subplots(figsize=(max(6, D * 1.2), 4))
    ax.boxplot(all_contractions, labels=param_names, patch_artist=True)
    ax.axhline(0.0, color="k",   linestyle="--", linewidth=1, label="no update")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="full contraction")
    ax.set_ylabel("Posterior contraction  (1 − σ_post / σ_prior)")
    ax.set_title(f"Posterior Update Check  (N={N_TEST} simulated datasets)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTDIR, "1_contraction_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    post0 = all_post_samples[0]   # (N_POST, D)
    ncols = min(D, 4)
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()
    for d, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(prior_samples[:, d].numpy(), bins=50, density=True,
                alpha=0.5, color="steelblue", label="prior")
        ax.hist(post0[:, d], bins=50, density=True,
                alpha=0.7, color="coral", label="posterior")
        ax.axvline(all_theta_true[0, d], color="k", linestyle="--", label="true")
        ax.set_title(name)
        ax.legend(fontsize=7)
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle("Prior vs Posterior (session 0)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "2_prior_vs_posterior.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    theta_true_pt = torch.from_numpy(all_theta_true[0:1])
    fig, _ = pairplot(
        torch.from_numpy(post0),
        points=theta_true_pt,
        labels=param_names,
        points_colors="r",
    )
    fig.suptitle("Posterior pairplot (session 0)", fontsize=11)
    path = os.path.join(OUTDIR, "3_pairplot_session0.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    summary_path = os.path.join(OUTDIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Posterior Update Check Summary\n")
        f.write(f"==============================\n")
        f.write(f"Model: {MODEL_NAME}  N_test: {N_TEST}  N_post: {N_POST}\n\n")
        f.write(f"{'param':>10}  {'prior_std':>10}  {'post_std_mean':>14}  {'contraction':>12}\n")
        for d, name in enumerate(param_names):
            f.write(
                f"{name:>10}  {prior_std[d]:>10.4f}  "
                f"{all_post_stds[:, d].mean():>14.4f}  "
                f"{all_contractions[:, d].mean():>12.4f}\n"
            )
    print("Saved:", summary_path)
    print("\nMean contraction per parameter:")
    for d, name in enumerate(param_names):
        print(f"  {name:10s}: {all_contractions[:, d].mean():.3f}")


if __name__ == "__main__":
    main()
