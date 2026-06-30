"""Simulation-Based Calibration (SBC) for a pretrained NPE model."""
from __future__ import annotations

import os
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.model_specs import select_model
from sbi_for_diffusion_models.mnpe import load_npe, run_sbc_npe

MODEL_NAME = os.environ.get("MODEL_NAME", "lapse")
MODEL_FILE = os.environ.get("MODEL_FILE", f"npe_rt_choice_{MODEL_NAME}.pt")


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    simulate_batch_fn, prior_theta, param_names, _model_tag, autoregressive = select_model(MODEL_NAME)

    if hasattr(prior_theta, "to"):
        prior_theta.to(torch.device(device))

    model_path = os.path.expanduser(os.path.join("~/models", MODEL_FILE))
    print(f"Loading model from {model_path} (ar={autoregressive}) ...")

    density_estimator, saved_cfg = load_npe(
        model_path,
        prior_theta=prior_theta,
        device=device,
    )
    saved_ar = bool(getattr(saved_cfg, "AUTOREGRESSIVE", False))
    if saved_ar != autoregressive:
        raise RuntimeError(
            f"MODEL_NAME={MODEL_NAME!r} implies ar={autoregressive}, "
            f"but checkpoint was trained with AUTOREGRESSIVE={saved_ar}."
        )

    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    outdir = os.environ.get("OUTDIR", f"sbc_outputs_{MODEL_NAME}")

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
