from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Task timescale (species-dependent)
#
# T_MAX and PULSE_INTERVAL are module-level globals read at call-time across the
# simulator, loaders, and embedding-net sizing (P = T_MAX / PULSE_INTERVAL).
# They are NOT stored per-run in the dataclass historically, so a model trained
# for one species must be loaded with the matching timescale in the environment.
#
# Presets (set SPECIES=marmoset|human|mouse), overridable individually:
#   marmoset/human : 4 Hz schedule, 250 ms pulses, 10 s cap  -> P = 40
#   mouse          : 10 Hz schedule, 100 ms pulses, 5 s cap   -> P = 50
#
# Explicit env vars T_MAX / PULSE_INTERVAL / P_SUCCESS_TRAIN_VALUES override the
# preset. With nothing set, defaults are the marmoset values (backward compatible).
# ---------------------------------------------------------------------------

_SPECIES_PRESETS = {
    "marmoset": dict(T_MAX=10.0, PULSE_INTERVAL=0.25,
                     P_SUCCESS_TRAIN_VALUES=(1.0, 0.9, 0.8, 0.7, 0.6)),
    "human":    dict(T_MAX=10.0, PULSE_INTERVAL=0.25,
                     P_SUCCESS_TRAIN_VALUES=(1.0, 0.9, 0.8, 0.7, 0.6)),
    "mouse":    dict(T_MAX=5.0,  PULSE_INTERVAL=0.10,
                     P_SUCCESS_TRAIN_VALUES=(1.0, 0.9, 0.8)),
}

SPECIES = os.environ.get("SPECIES", "marmoset").strip().lower()
if SPECIES not in _SPECIES_PRESETS:
    raise ValueError(
        f"Unknown SPECIES={SPECIES!r}; options: {sorted(_SPECIES_PRESETS)}"
    )
_preset = _SPECIES_PRESETS[SPECIES]


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else float(default)


def _env_p_success(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    val = os.environ.get(name)
    if val in (None, ""):
        return tuple(default)
    return tuple(float(v) for v in val.replace(" ", "").split(",") if v)


DT = 1e-6
DT_CHOICE = 5e-4
DT_INTERNAL = 0.01
FLASH_DURATION = 0.020

T_MAX = _env_float("T_MAX", _preset["T_MAX"])
PULSE_INTERVAL = _env_float("PULSE_INTERVAL", _preset["PULSE_INTERVAL"])

MAX_TIMEOUT_TRIES = 20
TIMEOUT_FRAC_ALLOWED = 0.20


@dataclass(frozen=True)
class RunConfig:
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.7
    P_SUCCESS_TRAIN_VALUES: Tuple[float, ...] = field(
        default_factory=lambda: _env_p_success(
            "P_SUCCESS_TRAIN_VALUES", _preset["P_SUCCESS_TRAIN_VALUES"]
        )
    )

    # Task timescale recorded for checkpoint provenance (mirrors module globals
    # at construction time). Fit scripts can assert these match the loaded
    # environment so a mouse checkpoint is never silently fed a marmoset P.
    SPECIES: str = SPECIES
    T_MAX: float = T_MAX
    PULSE_INTERVAL: float = PULSE_INTERVAL

    LOG_RT_MANUALLY: bool = True
    THETA_TRUE_FROM_PRIOR: bool = True

    NUM_TRIALS_OBS: int = field(
        default_factory=lambda: int(os.environ.get("NUM_TRIALS_OBS", "512"))
    )
    NPE_HIDDEN_FEATURES: int = 256
    NPE_NUM_TRANSFORMS: int = 10
    NPE_NUM_BINS: int = 12
    NPE_SESSIONS_PER_STEP: int = 256
    NPE_NUM_STEPS: int = 10000
    NPE_LR: float = 3e-4

    NPE_TRIAL_NET_HIDDEN: int = 256
    NPE_TRIAL_NET_LAYERS: int = 4
    NPE_TRIAL_NET_OUTPUT_DIM: int = 128
    NPE_AGG_FN: str = "mean"
    NPE_POST_AGG_HIDDEN: int = 256
    NPE_POST_AGG_LAYERS: int = 3
    NPE_EMBEDDING_OUTPUT_DIM: int = 128

    NPE_VAL_SESSIONS: int = 2048
    NPE_VAL_EVERY: int = 100
    NPE_VAL_PATIENCE: int = 20
    NPE_VAL_MIN_DELTA: float = 1e-3
    NPE_VAL_BATCH: int = 512
    NPE_VAL_SEED_OFFSET: int = 999_983

    NPE_POSTERIOR_SAMPLES: int = 20000

    RUN_SBC: bool = True
    NPE_SBC_NUM_DATASETS: int = 200
    NPE_SBC_POST_SAMPLES: int = 200

    AUTOREGRESSIVE: bool = False


RUN_CONFIG_PARAMS = RunConfig()
