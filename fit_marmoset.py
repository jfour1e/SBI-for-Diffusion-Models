"""
Fit the pretrained lapse NPE to real marmoset behavioral data.

Loads Helios (or any animal) 70-30 sessions, converts flash sequences to
the model's pulse format, samples the posterior for each session, and plots.

Usage
-----
  python fit_marmoset.py
  ANIMAL=Carter python fit_marmoset.py
  MODEL_NAME=base ANIMAL=Helios python fit_marmoset.py
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
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding
from sbi_for_diffusion_models.load_marmoset import load_marmoset_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

ANIMAL      = os.environ.get("ANIMAL",     "Tortellini")
STAGE       = os.environ.get("STAGE",      "70-30")
MODEL_NAME  = os.environ.get("MODEL_NAME", "lapse")
N_POST      = int(os.environ.get("N_POST",   "5000"))
N_PRIOR     = int(os.environ.get("N_PRIOR",  "5000"))
SEED        = int(os.environ.get("SEED",     "0"))
OUTDIR      = os.environ.get("OUTDIR",      f"marmoset_outputs/{ANIMAL}_{STAGE.replace('-','_')}")
MODEL_DIR   = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models"))
DATA_PATH   = os.environ.get(
    "DATA_PATH",
    "/projectnb/ssmsvi/rsenne/data_marmoset/marmoset_data.csv.gz",
)

def get_spec(model_name: str) -> dict:
    if model_name == "base":
        return dict(
            prior_builder=build_prior_theta,
            param_names=["a0", "lam", "v", "B", "tau"],
            model_file="npe_rt_choice_base.pt",
        )
    elif model_name == "lapse":
        return dict(
            prior_builder=build_prior_theta_lapse,
            param_names=["a0", "lam", "v", "B", "tau", "p_lapse"],
            model_file="npe_rt_choice_lapse.pt",
        )
    raise ValueError(model_name)

def load_npe(model_path: str, prior_theta, device: str):
    """Load the NPE and decouple the embedding net so we can feed variable-length sessions.

    Returns the density estimator (with Identity embedding), the original
    embedding net (accepts any trial count), the saved config, and T used
    during training.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg  = checkpoint["config"]
    dev        = torch.device(device)

    P         = max_num_pulses()
    T         = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P
    theta_dim = int(torch.as_tensor(prior_theta.sample((1,))).reshape(-1).numel())

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

    # Extract the trained embedding net and replace it with Identity so that
    # DirectPosterior skips the shape check on the raw x.  We will pre-embed
    # variable-length sessions ourselves before calling posterior.sample().
    embedding_net = de.net._embedding_net          # trained PermutationInvariantEmbedding
    emb_out_dim   = int(saved_cfg.NPE_EMBEDDING_OUTPUT_DIM)
    de.net._embedding_net = torch.nn.Identity()
    de._condition_shape   = torch.Size([emb_out_dim])

    return de, embedding_net, saved_cfg, T

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev    = torch.device(device)
    print(f"Animal: {ANIMAL}  Stage: {STAGE}  Model: {MODEL_NAME}  device: {device}")

    spec        = get_spec(MODEL_NAME)
    prior_theta = spec["prior_builder"]()
    param_names = spec["param_names"]
    D           = len(param_names)

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    model_path = os.path.join(MODEL_DIR, spec["model_file"])
    print(f"Loading NPE from {model_path} ...")
    de, embedding_net, saved_cfg, T = load_npe(model_path, prior_theta, device)

    posterior = DirectPosterior(
        posterior_estimator=de,
        prior=prior_theta,
        device=device,
    )

    sessions, meta = load_marmoset_sessions(
        csv_path=DATA_PATH,
        animal=ANIMAL,
        stage=STAGE,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=SEED,
    )
    if not sessions:
        raise RuntimeError(f"No valid sessions found for {ANIMAL} / {STAGE}")

    prior_samples = torch.as_tensor(
        prior_theta.sample((N_PRIOR,)), dtype=torch.float32
    ).cpu().numpy()

    os.makedirs(OUTDIR, exist_ok=True)

    # ── Combined posterior (all trials from all sessions) ──
    P = max_num_pulses()
    trial_dim = 2 + P
    all_trials = torch.cat(
        [x_flat.reshape(-1, trial_dim) for x_flat in sessions], dim=0
    )  # (total_trials, trial_dim)
    total_n = all_trials.shape[0]
    x_combined = all_trials.reshape(1, -1).to(device, dtype=torch.float32)

    print(f"\n[Combined]  {total_n} trials across {len(sessions)} sessions")
    with torch.no_grad():
        x_emb = embedding_net(x_combined)
    combined_samples = posterior.sample(
        (N_POST,), x=x_emb, show_progress_bars=True
    ).detach().cpu().numpy()

    np.save(os.path.join(OUTDIR, "posterior_combined.npy"), combined_samples)

    fig, _ = pairplot(
        torch.from_numpy(combined_samples),
        labels=param_names,
    )
    fig.suptitle(
        f"{ANIMAL} | {STAGE} | combined ({total_n} trials, {len(sessions)} sessions)",
        fontsize=10,
    )
    fig_path = os.path.join(OUTDIR, "pairplot_combined.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", fig_path)

    # ── Per-session posteriors ──
    all_samples = []

    for s_idx, (x_flat, m) in enumerate(zip(sessions, meta)):
        sess_label = str(m["session_datetime"]).replace(" ", "_").replace(":", "-")
        print(f"\n[Session {s_idx+1}/{len(sessions)}]  {sess_label}  "
              f"n={m['n_trials']}  acc={m['accuracy']:.3f}")

        x_flat = x_flat.to(device, dtype=torch.float32)
        with torch.no_grad():
            x_emb = embedding_net(x_flat)
        samples = posterior.sample(
            (N_POST,), x=x_emb, show_progress_bars=True
        ).detach().cpu().numpy()
        all_samples.append(samples)

        npy_path = os.path.join(OUTDIR, f"posterior_sess{s_idx+1}_{sess_label}.npy")
        np.save(npy_path, samples)

        fig, _ = pairplot(
            torch.from_numpy(samples),
            labels=param_names,
        )
        fig.suptitle(
            f"{ANIMAL} | {STAGE} | session {s_idx+1}\n"
            f"n={m['n_trials']}  acc={m['accuracy']:.3f}  "
            f"median_rt={m['rt_median']:.2f}s",
            fontsize=10,
        )
        fig_path = os.path.join(OUTDIR, f"pairplot_sess{s_idx+1}_{sess_label}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved:", fig_path)

    # ── Overlay: prior vs combined vs per-session ──
    ncols = min(D, 4)
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for d, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(prior_samples[:, d], bins=60, density=True,
                alpha=0.4, color="steelblue", label="prior")
        ax.hist(combined_samples[:, d], bins=60, density=True,
                alpha=0.6, color="black", label="combined",
                histtype="step", linewidth=2)
        for s_idx, samp in enumerate(all_samples):
            ax.hist(samp[:, d], bins=60, density=True,
                    alpha=0.5, label=f"sess {s_idx+1}", histtype="step", linewidth=1)
        ax.set_title(name)
        ax.legend(fontsize=7)

    for ax in axes[D:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{ANIMAL} {STAGE} — Prior vs Combined vs Per-Session Posteriors\n"
        f"({len(sessions)} sessions, {total_n} total trials)",
        fontsize=12,
    )
    fig.tight_layout()
    overlay_path = os.path.join(OUTDIR, "prior_vs_posteriors_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", overlay_path)

    summary_path = os.path.join(OUTDIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Marmoset Posterior Fit Summary\n")
        f.write(f"==============================\n")
        f.write(f"Animal: {ANIMAL}  Stage: {STAGE}  Model: {MODEL_NAME}\n")
        f.write(f"N sessions: {len(sessions)}  Total trials: {total_n}\n\n")
        f.write(f"Combined ({total_n} trials across all sessions)\n")
        for d, name in enumerate(param_names):
            f.write(f"  {name:10s}: mean={combined_samples[:,d].mean():.4f}  "
                    f"std={combined_samples[:,d].std():.4f}  "
                    f"[{np.percentile(combined_samples[:,d],5):.4f}, "
                    f"{np.percentile(combined_samples[:,d],95):.4f}]\n")
        f.write("\n")
        for s_idx, (samp, m) in enumerate(zip(all_samples, meta)):
            f.write(f"Session {s_idx+1}  ({m['session_datetime']})  "
                    f"n={m['n_trials']}  acc={m['accuracy']:.3f}\n")
            for d, name in enumerate(param_names):
                f.write(f"  {name:10s}: mean={samp[:,d].mean():.4f}  "
                        f"std={samp[:,d].std():.4f}  "
                        f"[{np.percentile(samp[:,d],5):.4f}, "
                        f"{np.percentile(samp[:,d],95):.4f}]\n")
            f.write("\n")
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
