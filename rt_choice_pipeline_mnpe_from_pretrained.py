"""Run inference from a pretrained NPE model (streaming-compatible architecture)."""
import os
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.priors import build_prior_theta, build_prior_theta_lapse
from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    max_num_pulses,
)
from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
)
from sbi_for_diffusion_models.mnpe import load_npe
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

MODEL_NAME = "lapse"

def get_model_spec(model_name: str):
    """
    Return model-specific objects needed for pretrained inference.
    """
    if model_name == "base":
        return {
            "prior_builder": build_prior_theta,
            "simulate_batch_fn": simulate_rt_choice_batch,
            "param_names": ("a0", "lam", "v", "B", "tau"),
            "model_filename": "npe_rt_choice_base.pt",
            "outdir_default": "npe_outputs_base",
        }
    elif model_name == "lapse":
        return {
            "prior_builder": build_prior_theta_lapse,
            "simulate_batch_fn": simulate_rt_choice_batch_lapse,
            "param_names": ("a0", "lam", "v", "B", "tau", "p_lapse"),
            "model_filename": "npe_rt_choice_lapse.pt",
            "outdir_default": "npe_outputs_lapse",
        }
    else:
        raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use 'base' or 'lapse'.")

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print("device:", device)

    spec = get_model_spec(MODEL_NAME)
    prior_theta = spec["prior_builder"]()
    simulate_batch_fn = spec["simulate_batch_fn"]
    param_names = spec["param_names"]

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    model_path = os.path.expanduser(os.path.join("~/models", spec["model_filename"]))
    print(f"Loading model from {model_path} ...")

    density_estimator, saved_cfg = load_npe(
        model_path,
        prior_theta=prior_theta,
        device=device,
    )

    print("Model loaded.")
    print("Estimator param device:", next(density_estimator.parameters()).device)

    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P

    # --- simulate a "true" theta and an observed session in exactly the same format as training ---
    if bool(saved_cfg.THETA_TRUE_FROM_PRIOR):
        theta_true = torch.as_tensor(
            prior_theta.sample((1,)),
            device=dev,
            dtype=torch.float32,
        ).reshape(-1)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    # simulate_training_sessions returns x already flattened as (N, T*(2+P+1)) with mask included
    _, x_o_flat = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=1,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(saved_cfg.MU_SENSORY),
        p_success=float(saved_cfg.P_SUCCESS),
        P=P,
        log_rt=bool(saved_cfg.LOG_RT_MANUALLY),
        seed=123,
        theta=theta_true,
    )
    x_3d = x_o_flat.view(1, T, trial_dim)[0]
    rt_valid = x_3d[:, 0]
    choice_valid = x_3d[:, 1].long()

    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())
    print("x_o_flat shape:", tuple(x_o_flat.shape), "device:", x_o_flat.device)
    print(f"num trials: {T}")
    print("rt[min,max]:", float(rt_valid.min()), float(rt_valid.max()))
    print("choice counts:", torch.bincount(choice_valid, minlength=2).tolist())

    # --- sample posterior ---
    n_samps = int(cfg.NPE_POSTERIOR_SAMPLES)
    print(f"\n--- Sampling posterior n={n_samps} ---")
    samples = posterior.sample(
        (n_samps,),
        x=x_o_flat,
        show_progress_bars=True,
    ).detach().cpu()

    # --- save outputs ---
    outdir = os.environ.get("OUTDIR", spec["outdir_default"])
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

    fig, ax = pairplot(
        samples,
        points=theta_true.detach().cpu().reshape(1, -1),
        labels=list(param_names),
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()