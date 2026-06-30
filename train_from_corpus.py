#!/usr/bin/env python
"""Train NPE from a pre-simulated corpus on disk.

Reads `corpus/<MODEL_NAME>/chunk_*.pt` for training and `val_*.pt` (optional)
for validation. Architecture and routing match `train_mixed_npe.py`; only the
data path is different.

Env vars:
  MODEL_NAME           required, must match what produced the corpus
  SEED                 default 0
  NPE_NUM_STEPS        default 60000
  NPE_PATIENCE         default 200 (val patience)
  CHECKPOINT_EVERY     default 1000
  MODEL_DIR            output dir for the trained checkpoint
  CORPUS_ROOT          dir where `MODEL_NAME/` subdir lives (default ./corpus)
  RESUME_FROM          optional path to a previous checkpoint
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi_for_diffusion_models.model_specs import select_model, simulation_overrides_for
from sbi_for_diffusion_models.mnpe import train_npe_session_from_corpus
from sbi_for_diffusion_models.data_simulator import corpus_dir
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

MODEL_NAME = os.environ.get("MODEL_NAME", "lapse_noleak_ar")
SEED = int(os.environ.get("SEED", "0"))
NPE_NUM_STEPS = int(os.environ.get("NPE_NUM_STEPS", "60000"))
NPE_PATIENCE = int(os.environ.get("NPE_PATIENCE", "200"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "1000"))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/models"
)
CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "corpus")
RESUME_FROM = os.environ.get("RESUME_FROM", "") or None


def _model_path(cfg, model_tag: str, suffix: str = "") -> str:
    values = "_".join(f"{int(round(v * 100)):03d}" for v in cfg.P_SUCCESS_TRAIN_VALUES)
    name = f"npe_{model_tag}_mixed_{values}{suffix}.pt"
    return os.path.join(MODEL_DIR, name)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulate_batch_fn, prior_theta, param_names, model_tag, autoregressive = select_model(MODEL_NAME)

    overrides = simulation_overrides_for(MODEL_NAME)
    T_override = overrides.get("num_trials_per_session")
    cfg_kwargs = dict(
        NPE_NUM_STEPS=NPE_NUM_STEPS,
        NPE_VAL_PATIENCE=NPE_PATIENCE,
        AUTOREGRESSIVE=autoregressive,
    )
    if T_override is not None:
        cfg_kwargs["NUM_TRIALS_OBS"] = int(T_override)
    cfg = replace(RUN_CONFIG_PARAMS, **cfg_kwargs)

    final_path = _model_path(cfg, model_tag)
    latest_path = _model_path(cfg, model_tag, suffix="_latest")
    os.makedirs(MODEL_DIR, exist_ok=True)

    cdir = corpus_dir(CORPUS_ROOT, MODEL_NAME)
    val_dir = cdir  # val_*.pt files live alongside chunk_*.pt
    if not os.path.isdir(cdir):
        raise FileNotFoundError(
            f"Corpus dir not found: {cdir}. Run pre_simulate.py with the same MODEL_NAME first."
        )

    print(f"Device:           {device}")
    print(f"Model:            {MODEL_NAME} ({model_tag}, ar={autoregressive})")
    print(f"Train values:     {cfg.P_SUCCESS_TRAIN_VALUES}")
    print(f"Steps:            {cfg.NPE_NUM_STEPS} (val patience={cfg.NPE_VAL_PATIENCE})")
    print(f"Sessions/step:    {cfg.NPE_SESSIONS_PER_STEP}")
    print(f"LR:               {cfg.NPE_LR}")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} steps -> {latest_path}")
    print(f"Final path:       {final_path}")
    print(f"Corpus dir:       {cdir}")
    print(f"Resume from:      {RESUME_FROM or '(none)'}")

    t0 = time.time()
    density_estimator, posterior = train_npe_session_from_corpus(
        cfg,
        prior_theta,
        corpus_train_dir=cdir,
        corpus_val_dir=val_dir,
        device=device,
        seed=SEED,
        resume_from=RESUME_FROM,
        checkpoint_path=latest_path,
        checkpoint_every=CHECKPOINT_EVERY,
        autoregressive=autoregressive,
    )
    elapsed = time.time() - t0
    print(f"\n[TRAIN] finished in {elapsed / 60:.1f} min")

    torch.save(
        {"state_dict": density_estimator.state_dict(), "config": cfg},
        final_path,
    )
    print(f"[TRAIN] Saved final: {final_path}")


if __name__ == "__main__":
    main()
