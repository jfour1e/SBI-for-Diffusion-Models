from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot

from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.mnpe import train_npe_session, run_inference_npe, run_sbc_npe
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions,
    simulate_observed_session,
    flatten_observed_session,
    summarize_trials,
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    P = max_num_pulses()
    trial_dim = 2 + P
    print(f"P = {P} pulses per trial, trial_dim = {trial_dim}")

    prior_theta = build_prior_theta()

    # ── 1. Simulate session-level training data ──
    print("\n--- Simulating training sessions ---")
    theta_train, x_train = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=cfg.NPE_NUM_SESSIONS,
        num_trials=cfg.NUM_TRIALS_OBS,
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        log_rt=cfg.LOG_RT_MANUALLY,
        seed=0,
    )

    # ── 2. Train NPE with permutation-invariant embedding ──
    print("\n--- Training NPE ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    density_estimator, inference_obj = train_npe_session(
        cfg, prior_theta, theta_train, x_train, device=device,
    )

    # Save model
    model_dir = os.path.expanduser("~/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "npe_rt_choice.pt")
    torch.save({
        "state_dict": density_estimator.state_dict(),
        "config": cfg,
    }, model_path)
    print("Saved NPE model to:", model_path)

    # ── 3. Simulate observed session ──
    if cfg.THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=cfg.NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # ── 4. Direct posterior sampling (no MCMC) ──
    x_o_flat = flatten_observed_session(x_o, pulses_o)
    print(f"\nFlattened observation shape: {tuple(x_o_flat.shape)}")

    print("\n--- Sampling posterior (direct, no MCMC) ---")
    samples = run_inference_npe(cfg, inference_obj, density_estimator, x_o_flat, prior_theta)

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

    # ── 6. SBC (optional — uncomment to run) ──
    # print("\n--- Running SBC ---")
    # run_sbc_npe(
    #     cfg,
    #     prior_theta=prior_theta,
    #     inference_obj=inference_obj,
    #     density_estimator=density_estimator,
    #     device="cpu",
    #     num_datasets=cfg.NPE_SBC_NUM_DATASETS,
    #     posterior_samples_per_dataset=cfg.NPE_SBC_POST_SAMPLES,
    #     seed=0,
    #     param_names=("a0", "lam", "v", "B", "tau"),
    #     outdir=os.path.join(outdir, "sbc"),
    #     plot_bins=30,
    # )


if __name__ == "__main__":
    main()
