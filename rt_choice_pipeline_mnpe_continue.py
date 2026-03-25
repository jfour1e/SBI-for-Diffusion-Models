"""Continue training an NPE from a saved checkpoint."""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot

from sbi_for_diffusion_models.priors import build_prior_theta_lapse
from sbi_for_diffusion_models.models.lapse_rt_choice_model import simulate_rt_choice_batch_lapse
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.mnpe import train_npe_session, run_sbc_npe
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

# Number of steps already completed in the previous run — shifts training
# seeds so we see fresh data instead of replaying the same batches.
STEPS_ALREADY_DONE = int(os.environ.get("STEPS_ALREADY_DONE", 2000))

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print("device:", device)

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P
    print(f"P={P}, T={T}, trial_dim={trial_dim}")

    prior_theta = build_prior_theta_lapse()
    simulate_batch_fn = simulate_rt_choice_batch_lapse
    param_names = ("a0", "lam", "v", "B", "tau", "p_lapse")
    model_tag = "lapse"

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    model_dir = os.path.expanduser("~/models")
    resume_path = os.path.join(model_dir, f"npe_rt_choice_{model_tag}.pt")
    print(f"Resuming from: {resume_path}  (seed_offset={STEPS_ALREADY_DONE})")

    print("\n--- Continuing NPE training ---")
    density_estimator, posterior_obj = train_npe_session(
        cfg,
        prior_theta,
        simulate_batch_fn=simulate_batch_fn,
        device=device,
        seed=0,
        resume_from=resume_path,
        seed_offset=STEPS_ALREADY_DONE,
    )

    # Overwrite checkpoint with improved model
    torch.save({"state_dict": density_estimator.state_dict(), "config": cfg}, resume_path)
    print("Saved NPE model to:", resume_path)

    # Quick posterior check on one simulated session
    print("\n--- Simulating observed session ---")
    theta_true = torch.as_tensor(
        prior_theta.sample((1,)), device=dev, dtype=torch.float32
    ).reshape(-1)

    _, x_o_flat = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=1,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=456,
        theta=theta_true,
    )

    outdir = os.environ.get("OUTDIR", f"npe_outputs_{model_tag}")
    os.makedirs(outdir, exist_ok=True)

    x_o_flat = x_o_flat.to(dev, dtype=torch.float32)
    samples = posterior_obj.sample(
        (int(cfg.NPE_POSTERIOR_SAMPLES),), x=x_o_flat, show_progress_bars=True
    ).detach().cpu()

    npy_path = os.path.join(outdir, "posterior_samples_theta_resumed.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

    fig, _ = pairplot(
        samples,
        points=theta_true.detach().cpu().reshape(1, -1),
        labels=list(param_names),
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta_resumed.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)

    # SBC
    if bool(getattr(cfg, "RUN_SBC", False)):
        print("\n--- Running SBC ---")
        run_sbc_npe(
            cfg,
            prior_theta=prior_theta,
            posterior=posterior_obj,
            simulate_batch_fn=simulate_batch_fn,
            device=device,
            num_datasets=int(cfg.NPE_SBC_NUM_DATASETS),
            posterior_samples_per_dataset=int(cfg.NPE_SBC_POST_SAMPLES),
            seed=1,
            param_names=param_names,
            outdir=os.path.join(outdir, "sbc_resumed"),
            plot_bins=30,
        )

if __name__ == "__main__":
    main()
