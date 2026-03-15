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
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

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

    prior_theta = build_prior_theta()
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    # train
    print("\n--- Training NPE ---")
    density_estimator, posterior_obj = train_npe_session(
        cfg, prior_theta, device=device, seed=0
    )
    print("density_estimator device:", next(density_estimator.parameters()).device)

    model_dir = os.path.expanduser("~/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "npe_rt_choice.pt")
    torch.save({"state_dict": density_estimator.state_dict(), "config": cfg}, model_path)
    print("Saved NPE model to:", model_path)

    # simulate observed session
    print("\n--- Simulating observed session ---")
    if bool(cfg.THETA_TRUE_FROM_PRIOR):
        theta_true = prior_theta.sample((1,)).view(5).to(device=dev, dtype=torch.float32)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    _, x_o_flat = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=1,
        num_trials=T,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=123,
        theta=theta_true,
    )

    x_3d = x_o_flat.view(1, T, trial_dim)[0]
    rt_valid = x_3d[:, 0]
    choice_valid = x_3d[:, 1].long()
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())
    print(f"num trials: {T}")
    print("rt[min,max]:", float(rt_valid.min()), float(rt_valid.max()))
    print("choice counts:", torch.bincount(choice_valid, minlength=2).tolist())

    # posterior sampling
    print("\n--- Sampling posterior ---")
    x_o_flat = x_o_flat.to(next(density_estimator.parameters()).device, dtype=torch.float32)
    samples = posterior_obj.sample(
        (int(cfg.NPE_POSTERIOR_SAMPLES),),
        x=x_o_flat,
        show_progress_bars=True,
    ).detach().cpu()
    
    # save outputs
    outdir = os.environ.get("OUTDIR", "npe_outputs")
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

    fig, ax = pairplot(
        samples,
        points=theta_true.detach().cpu().view(1, -1),
        labels=["a0", "lam", "v", "B", "tau"],
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)

    # SBC
    do_sbc = bool(getattr(cfg, "RUN_SBC", False))
    if do_sbc:
        print("\n--- Running SBC ---")
        sbc_dir = os.path.join(outdir, "sbc")
        run_sbc_npe(
            cfg,
            prior_theta=prior_theta,
            posterior=posterior_obj,
            device=device,
            num_datasets=int(cfg.NPE_SBC_NUM_DATASETS),
            posterior_samples_per_dataset=int(cfg.NPE_SBC_POST_SAMPLES),
            seed=0,
            param_names=("a0", "lam", "v", "B", "tau"),
            outdir=sbc_dir,
            plot_bins=30,
        )

if __name__ == "__main__":
    main()
