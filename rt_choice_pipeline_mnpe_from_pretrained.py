"""Run inference from a pretrained NPE model (no simulation or training)."""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets import posterior_nn

from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.mnpe import SessionEmbeddingNet, run_sbc_npe
from sbi_for_diffusion_models.data_simulator import (
    simulate_observed_session,
    flatten_observed_session,
    summarize_trials,
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

MODEL_PATH = os.path.expanduser("~/models/npe_rt_choice.pt")


def load_npe(model_path: str, device: str = "cpu"):
    """Rebuild the NPE architecture and load saved weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    P = max_num_pulses()
    trial_dim = 2 + P
    num_trials = saved_cfg.NUM_TRIALS_OBS

    embedding_net = SessionEmbeddingNet(
        num_trials=num_trials,
        trial_dim=trial_dim,
        trial_net_hidden=saved_cfg.NPE_TRIAL_NET_HIDDEN,
        trial_net_layers=saved_cfg.NPE_TRIAL_NET_LAYERS,
        trial_net_output_dim=saved_cfg.NPE_TRIAL_NET_OUTPUT_DIM,
        aggregation_fn=saved_cfg.NPE_AGG_FN,
        post_agg_hidden=saved_cfg.NPE_POST_AGG_HIDDEN,
        post_agg_layers=saved_cfg.NPE_POST_AGG_LAYERS,
        output_dim=saved_cfg.NPE_EMBEDDING_OUTPUT_DIM,
    )

    # Build the same flow architecture used during training
    est_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=saved_cfg.NPE_HIDDEN_FEATURES,
        num_transforms=saved_cfg.NPE_NUM_TRANSFORMS,
        num_bins=saved_cfg.NPE_NUM_BINS,
        embedding_net=embedding_net,
    )

    # est_builder is a callable; we need to instantiate it with dummy data
    # to get the actual nn.Module, then load the state dict.
    theta_dim = 5
    x_dim = num_trials * trial_dim
    dummy_theta = torch.randn(2, theta_dim)
    dummy_x = torch.randn(2, x_dim)
    density_estimator = est_builder(dummy_theta, dummy_x)
    density_estimator.load_state_dict(checkpoint["state_dict"])
    density_estimator.to(device)
    density_estimator.eval()

    return density_estimator, saved_cfg


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load pretrained model ──
    print(f"Loading model from {MODEL_PATH} ...")
    density_estimator, saved_cfg = load_npe(MODEL_PATH, device=device)
    print("Model loaded.")

    P = max_num_pulses()
    prior_theta = build_prior_theta()

    # Move prior to model device so DirectPosterior is happy
    if device != "cpu" and hasattr(prior_theta, "to"):
        prior_theta.to(device)

    # ── 2. Build posterior directly (no inference object needed) ──
    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    # ── 3. Simulate observed session ──
    if saved_cfg.THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5).cpu()
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=saved_cfg.NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=saved_cfg.MU_SENSORY,
        p_success=saved_cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=saved_cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # ── 4. Direct posterior sampling ──
    x_o_flat = flatten_observed_session(x_o, pulses_o).to(device)
    print(f"\nFlattened observation shape: {tuple(x_o_flat.shape)}")

    print("\n--- Sampling posterior (direct, no MCMC) ---")
    samples = posterior.sample(
        (saved_cfg.NPE_POSTERIOR_SAMPLES,),
        x=x_o_flat,
        show_progress_bars=True,
    ).detach().cpu()

    # ── 5. Save outputs ──
    outdir = os.environ.get("OUTDIR", "npe_outputs")
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

    fig, ax = pairplot(
        samples,
        points=theta_true.view(1, -1).cpu(),
        labels=["a0", "lam", "v", "B", "tau"],
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)

    # ── 6. SBC ──
    print("\n--- Running SBC ---")
    sbc_outdir = os.path.join(outdir, "sbc")
    run_sbc_npe(
        saved_cfg,
        prior_theta=prior_theta,
        posterior=posterior,
        device=device,
        num_datasets=saved_cfg.NPE_SBC_NUM_DATASETS,
        posterior_samples_per_dataset=saved_cfg.NPE_SBC_POST_SAMPLES,
        seed=0,
        param_names=("a0", "lam", "v", "B", "tau"),
        outdir=sbc_outdir,
        plot_bins=30,
    )


if __name__ == "__main__":
    main()
