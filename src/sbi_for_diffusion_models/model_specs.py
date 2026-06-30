"""Centralized model registry for the RT-choice family.

One place to map a MODEL_NAME string -> (simulator, prior, param_names, ar).
All training/inference/analysis scripts should `select_model(MODEL_NAME)` here
so that adding a variant only touches this file.
"""
from __future__ import annotations

from typing import Callable, Tuple

from .priors import (
    build_prior_theta,
    build_prior_theta_lapse,
    build_prior_theta_noleak,
    build_prior_theta_lapse_noleak,
    build_prior_theta_ar,
    build_prior_theta_noleak_ar,
    build_prior_theta_lapse_ar,
    build_prior_theta_lapse_noleak_ar,
    build_prior_theta_lapse_noleak_mouse,
    build_prior_theta_lapse_noleak_ar_mouse,
)
from .models.rt_choice_model import (
    simulate_rt_choice_batch,
    simulate_rt_choice_batch_noleak,
    simulate_rt_choice_batch_ar,
    simulate_rt_choice_batch_noleak_ar,
    generate_pulses_torch,
    generate_pulses_human_torch,
    sample_p_success_human_cascade,
)
from .models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
    simulate_rt_choice_batch_lapse_noleak,
    simulate_rt_choice_batch_lapse_ar,
    simulate_rt_choice_batch_lapse_noleak_ar,
)


_MODEL_SPECS: dict[str, tuple[Callable, Callable, Tuple[str, ...], bool]] = {
    "base": (
        simulate_rt_choice_batch,
        build_prior_theta,
        ("a0", "lam", "v", "B", "tau"),
        False,
    ),
    "lapse": (
        simulate_rt_choice_batch_lapse,
        build_prior_theta_lapse,
        ("a0", "lam", "v", "B", "tau", "p_lapse"),
        False,
    ),
    "base_noleak": (
        simulate_rt_choice_batch_noleak,
        build_prior_theta_noleak,
        ("a0", "v", "B", "tau"),
        False,
    ),
    "lapse_noleak": (
        simulate_rt_choice_batch_lapse_noleak,
        build_prior_theta_lapse_noleak,
        ("a0", "v", "B", "tau", "p_lapse"),
        False,
    ),
    "base_ar": (
        simulate_rt_choice_batch_ar,
        build_prior_theta_ar,
        ("a0", "lam", "v", "B", "tau", "w_corr", "w_err"),
        True,
    ),
    "lapse_ar": (
        simulate_rt_choice_batch_lapse_ar,
        build_prior_theta_lapse_ar,
        ("a0", "lam", "v", "B", "tau", "p_lapse", "w_corr", "w_err"),
        True,
    ),
    "base_noleak_ar": (
        simulate_rt_choice_batch_noleak_ar,
        build_prior_theta_noleak_ar,
        ("a0", "v", "B", "tau", "w_corr", "w_err"),
        True,
    ),
    "lapse_noleak_ar": (
        simulate_rt_choice_batch_lapse_noleak_ar,
        build_prior_theta_lapse_noleak_ar,
        ("a0", "v", "B", "tau", "p_lapse", "w_corr", "w_err"),
        True,
    ),
    # Same architecture and prior as lapse_noleak_ar, but uses the human task's
    # independent-Bernoulli pulse generator and the coin-flip cascade for
    # per-trial p_R. Task-specific overrides (T=200, generators) are looked up
    # via `simulation_overrides_for(model_name)` below.
    "lapse_noleak_ar_human": (
        simulate_rt_choice_batch_lapse_noleak_ar,
        build_prior_theta_lapse_noleak_ar,
        ("a0", "v", "B", "tau", "p_lapse", "w_corr", "w_err"),
        True,
    ),
    # Non-AR human counterpart of lapse_noleak_ar_human: same human task
    # (independent-Bernoulli pulses, coin-flip cascade p_R, T=200) but no
    # history term. Enables the human AR-vs-non-AR model comparison.
    "lapse_noleak_human": (
        simulate_rt_choice_batch_lapse_noleak,
        build_prior_theta_lapse_noleak,
        ("a0", "v", "B", "tau", "p_lapse"),
        False,
    ),
    # Mouse task variants. Same simulators as the marmoset no-leak lapse models
    # (exclusive XOR pulses, the default generate_pulses_torch), but with
    # mouse-retuned priors. The task timescale (PULSE_INTERVAL=0.1, T_MAX=5 ->
    # P=50) and P_SUCCESS_TRAIN_VALUES=(1.0,0.9,0.8) come from SPECIES=mouse in
    # run_config, which must be set in the environment at train AND fit time.
    "lapse_noleak_mouse": (
        simulate_rt_choice_batch_lapse_noleak,
        build_prior_theta_lapse_noleak_mouse,
        ("a0", "v", "B", "tau", "p_lapse"),
        False,
    ),
    "lapse_noleak_ar_mouse": (
        simulate_rt_choice_batch_lapse_noleak_ar,
        build_prior_theta_lapse_noleak_ar_mouse,
        ("a0", "v", "B", "tau", "p_lapse", "w_corr", "w_err"),
        True,
    ),
}

MODEL_NAMES = tuple(_MODEL_SPECS.keys())


# Task-specific simulation overrides keyed by model_name. Only entries that
# differ from the defaults are listed; missing keys fall back to the marmoset
# defaults (NUM_TRIALS_OBS from RunConfig, generate_pulses_torch, no per-trial
# p_R sampler).
_SIMULATION_OVERRIDES: dict[str, dict] = {
    "lapse_noleak_ar_human": {
        "num_trials_per_session": 200,        # humans do one ~200-trial session
        "pulse_generator_fn": generate_pulses_human_torch,
        "p_success_per_trial_fn": sample_p_success_human_cascade,
    },
    "lapse_noleak_human": {
        "num_trials_per_session": 200,
        "pulse_generator_fn": generate_pulses_human_torch,
        "p_success_per_trial_fn": sample_p_success_human_cascade,
    },
}


def simulation_overrides_for(model_name: str) -> dict:
    """Return task-specific simulation overrides for `model_name`.

    Keys (all optional):
      num_trials_per_session: int   override cfg.NUM_TRIALS_OBS for this model
      pulse_generator_fn: Callable  used in place of generate_pulses_torch
      p_success_per_trial_fn: Callable  sample fresh p_R per trial (vs per session)
    """
    return dict(_SIMULATION_OVERRIDES.get(model_name, {}))


def select_model(model_name: str):
    """Return (simulate_batch_fn, prior_theta, param_names, model_tag, autoregressive)."""
    if model_name not in _MODEL_SPECS:
        raise ValueError(
            f"Unknown MODEL_NAME={model_name!r}. Options: {list(MODEL_NAMES)}"
        )
    sim_fn, prior_builder, param_names, ar = _MODEL_SPECS[model_name]
    return sim_fn, prior_builder(), param_names, model_name, ar


def model_spec(model_name: str):
    """Return the raw spec tuple (sim_fn, prior_builder, param_names, ar) without instantiating prior."""
    if model_name not in _MODEL_SPECS:
        raise ValueError(
            f"Unknown MODEL_NAME={model_name!r}. Options: {list(MODEL_NAMES)}"
        )
    return _MODEL_SPECS[model_name]
