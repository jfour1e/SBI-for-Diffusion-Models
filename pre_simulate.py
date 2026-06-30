#!/usr/bin/env python
"""Pre-simulate one chunk of (theta, x) pairs to disk.

Designed for SCC array jobs: each task index becomes one chunk. With AR models
the simulator is sequential-in-T, so spreading chunk generation across many
nodes is much faster than online simulation inside the training loop.

Env vars:
  MODEL_NAME           required, e.g. lapse_noleak_ar
  SEED                 base seed (default 0); per-chunk seed = SEED + chunk_idx
  CORPUS_ROOT          dir to write under (default ./corpus)
  N_SESSIONS           sessions per chunk (default 2048)
  KIND                 "train" or "val" (default "train")
  CHUNK_INDEX          override chunk index (default $SGE_TASK_ID, then 1)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi_for_diffusion_models.model_specs import select_model, simulation_overrides_for
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import (
    simulate_chunk_to_disk,
    corpus_dir,
    chunk_filename,
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS


def _resolve_chunk_index() -> int:
    if "CHUNK_INDEX" in os.environ:
        return int(os.environ["CHUNK_INDEX"])
    if "SGE_TASK_ID" in os.environ and os.environ["SGE_TASK_ID"] != "undefined":
        return int(os.environ["SGE_TASK_ID"])
    return 1


def main():
    MODEL_NAME = os.environ["MODEL_NAME"]
    SEED_BASE = int(os.environ.get("SEED", "0"))
    CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "corpus")
    N_SESSIONS = int(os.environ.get("N_SESSIONS", "2048"))
    KIND = os.environ.get("KIND", "train").lower()
    if KIND not in {"train", "val"}:
        raise ValueError(f"KIND must be 'train' or 'val'; got {KIND!r}")

    chunk_idx = _resolve_chunk_index()
    seed = SEED_BASE + chunk_idx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    simulate_batch_fn, prior_theta, param_names, _model_tag, autoregressive = select_model(MODEL_NAME)
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    cfg = RUN_CONFIG_PARAMS
    P = max_num_pulses()
    overrides = simulation_overrides_for(MODEL_NAME)
    T = int(overrides.get("num_trials_per_session", cfg.NUM_TRIALS_OBS))
    pulse_generator_fn = overrides.get("pulse_generator_fn")
    p_success_per_trial_fn = overrides.get("p_success_per_trial_fn")

    out_dir = corpus_dir(CORPUS_ROOT, MODEL_NAME)
    out_path = os.path.join(out_dir, chunk_filename(chunk_idx, kind=KIND))

    print(f"=== pre_simulate ===")
    print(f"Model:     {MODEL_NAME} (ar={autoregressive}, params={param_names})")
    print(f"Device:    {device}")
    print(f"Chunk:     {KIND} idx={chunk_idx} seed={seed}")
    print(f"Sessions:  {N_SESSIONS}  Trials/session: {T}  P={P}")
    print(f"Out path:  {out_path}")
    if pulse_generator_fn is not None or p_success_per_trial_fn is not None:
        print(f"Overrides: pulse_generator={getattr(pulse_generator_fn, '__name__', None)}, "
              f"p_success_per_trial={getattr(p_success_per_trial_fn, '__name__', None)}")
    else:
        print(f"P_SUCCESS_TRAIN_VALUES: {cfg.P_SUCCESS_TRAIN_VALUES}")

    t0 = time.time()
    theta_dim, x_dim = simulate_chunk_to_disk(
        out_path,
        prior_theta=prior_theta,
        num_sessions=N_SESSIONS,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=cfg.P_SUCCESS_TRAIN_VALUES,
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=seed,
        autoregressive=autoregressive,
        cfg=cfg,
        pulse_generator_fn=pulse_generator_fn,
        p_success_per_trial_fn=p_success_per_trial_fn,
    )
    elapsed = time.time() - t0
    bytes_written = os.path.getsize(out_path)
    print(
        f"[done] {elapsed/60:.1f} min  "
        f"theta_dim={theta_dim} x_dim={x_dim}  "
        f"file={bytes_written/2**20:.1f} MB"
    )


if __name__ == "__main__":
    main()
