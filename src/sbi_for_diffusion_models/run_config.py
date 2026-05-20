from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

DT = 1e-6
DT_CHOICE = 5e-4
DT_INTERNAL = 0.01
FLASH_DURATION = 0.020
T_MAX = 10.0
PULSE_INTERVAL = 0.25  # 4 Hz schedule, 40 pulses over 10 s

MAX_TIMEOUT_TRIES = 20
TIMEOUT_FRAC_ALLOWED = 0.20


@dataclass(frozen=True)
class RunConfig:
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.7
    P_SUCCESS_TRAIN_VALUES: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6)

    LOG_RT_MANUALLY: bool = True
    THETA_TRUE_FROM_PRIOR: bool = True

    NUM_TRIALS_OBS: int = 512
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


RUN_CONFIG_PARAMS = RunConfig()
