#!/usr/bin/env python
"""Fit one amortized NPE network on a mixture of p_success conditions.

Train-only; consume the resulting checkpoint with an inference script that
calls `load_npe` (e.g. scripts/fit_all_marmosets_all_stages.py).

Environment overrides:
  MODEL_NAME, SEED, NPE_NUM_STEPS, NPE_PATIENCE, CHECKPOINT_EVERY,
  MODEL_DIR, RESUME_FROM, DO_SBC, SBC_OUTDIR.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi_for_diffusion_models.priors import build_prior_theta_lapse, build_prior_theta
from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
)
from sbi_for_diffusion_models.models.rt_choice_model import simulate_rt_choice_batch
from sbi_for_diffusion_models.mnpe import train_npe_session, run_sbc_npe
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

MODEL_NAME = os.environ.get("MODEL_NAME", "lapse")
SEED = int(os.environ.get("SEED", "0"))
NPE_NUM_STEPS = int(os.environ.get("NPE_NUM_STEPS", "60000"))
NPE_PATIENCE = int(os.environ.get("NPE_PATIENCE", "200"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "1000"))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/models"
)
RESUME_FROM = os.environ.get("RESUME_FROM", "") or None
DO_SBC = int(os.environ.get("DO_SBC", "0"))
SBC_OUTDIR = os.environ.get("SBC_OUTDIR", "sbc_npe_outputs/mixed")


def _select_model(model_name: str):
    if model_name == "base":
        return (
            simulate_rt_choice_batch,
            build_prior_theta(),
            ("a0", "lam", "v", "B", "tau"),
            "base",
        )
    if model_name == "lapse":
        return (
            simulate_rt_choice_batch_lapse,
            build_prior_theta_lapse(),
            ("a0", "lam", "v", "B", "tau", "p_lapse"),
            "lapse",
        )
    raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use 'base' or 'lapse'.")


def _mixed_model_path(cfg, model_tag: str, suffix: str = "") -> str:
    values = "_".join(f"{int(round(v * 100)):03d}" for v in cfg.P_SUCCESS_TRAIN_VALUES)
    name = f"npe_{model_tag}_mixed_{values}{suffix}.pt"
    return os.path.join(MODEL_DIR, name)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulate_batch_fn, prior_theta, param_names, model_tag = _select_model(MODEL_NAME)

    cfg = replace(
        RUN_CONFIG_PARAMS,
        NPE_NUM_STEPS=NPE_NUM_STEPS,
        NPE_VAL_PATIENCE=NPE_PATIENCE,
    )

    final_path = _mixed_model_path(cfg, model_tag)
    latest_path = _mixed_model_path(cfg, model_tag, suffix="_latest")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Device:           {device}")
    print(f"Model:            {MODEL_NAME} ({model_tag})")
    print(f"Train values:     {cfg.P_SUCCESS_TRAIN_VALUES}")
    print(f"Steps:            {cfg.NPE_NUM_STEPS} (val patience={cfg.NPE_VAL_PATIENCE})")
    print(f"Sessions/step:    {cfg.NPE_SESSIONS_PER_STEP}")
    print(f"LR:               {cfg.NPE_LR}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} steps -> {latest_path}")
    print(f"Final path:       {final_path}")
    print(f"Resume from:      {RESUME_FROM or '(none)'}")

    t0 = time.time()
    density_estimator, posterior = train_npe_session(
        cfg,
        prior_theta,
        simulate_batch_fn=simulate_batch_fn,
        device=device,
        seed=SEED,
        resume_from=RESUME_FROM,
        checkpoint_path=latest_path,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    elapsed = time.time() - t0
    print(f"\n[TRAIN] finished in {elapsed / 60:.1f} min")

    torch.save(
        {"state_dict": density_estimator.state_dict(), "config": cfg},
        final_path,
    )
    print(f"[TRAIN] Saved final: {final_path}")

    if DO_SBC:
        run_sbc_npe(
            cfg,
            prior_theta=prior_theta,
            posterior=posterior,
            simulate_batch_fn=simulate_batch_fn,
            device=device,
            num_datasets=int(cfg.NPE_SBC_NUM_DATASETS),
            posterior_samples_per_dataset=int(cfg.NPE_SBC_POST_SAMPLES),
            seed=SEED,
            param_names=param_names,
            outdir=SBC_OUTDIR,
            plot_bins=30,
        )


if __name__ == "__main__":
    main()
