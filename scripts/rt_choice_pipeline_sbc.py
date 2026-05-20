"""Simulation-Based Calibration (SBC) for a pretrained NPE model."""
from __future__ import annotations

import os
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.priors import build_prior_theta, build_prior_theta_lapse
from sbi_for_diffusion_models.models.rt_choice_model import simulate_rt_choice_batch
from sbi_for_diffusion_models.models.lapse_rt_choice_model import simulate_rt_choice_batch_lapse
from sbi_for_diffusion_models.mnpe import load_npe, run_sbc_npe

MODEL_NAME = os.environ.get("MODEL_NAME", "lapse")


def get_model_spec(model_name: str):
    if model_name == "base":
        return {
            "prior_builder": build_prior_theta,
            "simulate_batch_fn": simulate_rt_choice_batch,
            "param_names": ("a0", "lam", "v", "B", "tau"),
            "model_filename": "npe_rt_choice_base.pt",
            "outdir_default": "sbc_outputs_base",
        }
    if model_name == "lapse":
        return {
            "prior_builder": build_prior_theta_lapse,
            "simulate_batch_fn": simulate_rt_choice_batch_lapse,
            "param_names": ("a0", "lam", "v", "B", "tau", "p_lapse"),
            "model_filename": "npe_rt_choice_lapse.pt",
            "outdir_default": "sbc_outputs_lapse",
        }
    raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use 'base' or 'lapse'.")


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    spec = get_model_spec(MODEL_NAME)
    prior_theta = spec["prior_builder"]()
    simulate_batch_fn = spec["simulate_batch_fn"]
    param_names = spec["param_names"]

    if hasattr(prior_theta, "to"):
        prior_theta.to(torch.device(device))

    model_path = os.path.expanduser(os.path.join("~/models", spec["model_filename"]))
    print(f"Loading model from {model_path} ...")

    density_estimator, saved_cfg = load_npe(
        model_path,
        prior_theta=prior_theta,
        device=device,
    )

    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    outdir = os.environ.get("OUTDIR", spec["outdir_default"])

    run_sbc_npe(
        cfg=saved_cfg,
        prior_theta=prior_theta,
        posterior=posterior,
        simulate_batch_fn=simulate_batch_fn,
        device=device,
        num_datasets=int(getattr(saved_cfg, "NPE_SBC_NUM_DATASETS", 100)),
        posterior_samples_per_dataset=int(getattr(saved_cfg, "NPE_SBC_POST_SAMPLES", 500)),
        seed=123,
        param_names=param_names,
        outdir=outdir,
        plot_bins=30,
    )


if __name__ == "__main__":
    main()
