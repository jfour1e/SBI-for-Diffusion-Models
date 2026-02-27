from __future__ import annotations
from dataclasses import dataclass 

# defines constants used in the SBI for Diffusion Models project
DT = 1e-6
DT_CHOICE = 5e-4
DT_INTERNAL = 0.01
T_MAX = 10.0
PULSE_INTERVAL = 0.25 # in seconds (i.e., 250 ms)

# Refinement sub-steps used only inside the single crossing interval.
# resolution = PULSE_INTERVAL / _DEFAULT_N_REFINE  (default 0.1 ms)
_DEFAULT_N_REFINE = 1000

@dataclass(frozen=True)
class RunConfig:
# Data / simulator settings
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.6
    NUM_TRIALS_OBS : int = 5_000

    # We recommend log-transforming RT but NOT the categorical choice.
    LOG_RT_MANUALLY: bool = False
    THETA_TRUE_FROM_PRIOR: bool = True

    # NPE settings
    NPE_NUM_SESSIONS: int = 100_000
    NPE_TRAIN_BATCH_SIZE: int = 512
    NPE_HIDDEN_FEATURES: int = 128
    NPE_NUM_TRANSFORMS: int = 5
    NPE_NUM_BINS: int = 8
    NPE_SESSIONS_PER_STEP: int = 512
    NPE_NUM_STEPS: int = 2_000
    NPE_LR: float = 5e-4

    # Embedding network (DeepSets)
    NPE_TRIAL_NET_HIDDEN: int = 128
    NPE_TRIAL_NET_LAYERS: int = 3
    NPE_TRIAL_NET_OUTPUT_DIM: int = 64
    NPE_AGG_FN: str = "mean"
    NPE_POST_AGG_HIDDEN: int = 128
    NPE_POST_AGG_LAYERS: int = 2
    NPE_EMBEDDING_OUTPUT_DIM: int = 64

    # NPE inference
    NPE_POSTERIOR_SAMPLES: int = 20000

    # NPE SBC
    RUN_SBC: bool = False
    NPE_SBC_NUM_DATASETS: int = 5
    NPE_SBC_POST_SAMPLES: int = 200


RUN_CONFIG_PARAMS = RunConfig()