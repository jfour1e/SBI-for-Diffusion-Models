from __future__ import annotations
from dataclasses import dataclass 

# defines constants used in the SBI for Diffusion Models project
DT = 1e-6
DT_CHOICE = 5e-4
DT_INTERNAL = 0.01
FLASH_DURATION = 0.020     
T_MAX = 10.0
PULSE_INTERVAL = 0.25  # in seconds (250 ms, 4 Hz schedule, 40 pulses over 10 s)

# regeneration tolerance 
MAX_TIMEOUT_TRIES = 20
TIMEOUT_FRAC_ALLOWED = 0.20

@dataclass(frozen=True)
class RunConfig:
# Data / simulator settings
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.7

    # We recommend log-transforming RT but NOT the categorical choice.
    LOG_RT_MANUALLY: bool = True
    THETA_TRUE_FROM_PRIOR: bool = True

    # NPE settings
    NUM_TRIALS_OBS : int = 512
    NPE_NUM_SESSIONS: int = 20
    NPE_HIDDEN_FEATURES: int = 256   # flow hidden dim 
    NPE_NUM_TRANSFORMS: int = 10     # NSF coupling layers 
    NPE_NUM_BINS: int = 12           # spline bins per transform
    NPE_SESSIONS_PER_STEP: int = 256
    NPE_NUM_STEPS: int = 10000       # training steps
    NPE_LR: float = 3e-4             # slightly lower LR for larger model

    # Simulation Reservoir settings
    NPE_RESERVOIR_SIZE: int = 1_000  
    NPE_RESERVOIR_REFRESH_FRAC: float = 0.0  # percentages of reservoir sims to resample to prevent staleness

    # Embedding network (DeepSets)
    NPE_TRIAL_NET_HIDDEN: int = 256  # per-trial MLP width (was 128)
    NPE_TRIAL_NET_LAYERS: int = 4    # per-trial MLP depth (was 3)
    NPE_TRIAL_NET_OUTPUT_DIM: int = 128  # per-trial embedding dim (was 64)
    NPE_AGG_FN: str = "mean"
    NPE_POST_AGG_HIDDEN: int = 256   # post-aggregation MLP width (was 128)
    NPE_POST_AGG_LAYERS: int = 3     # post-aggregation MLP depth (was 2)
    NPE_EMBEDDING_OUTPUT_DIM: int = 128  # session embedding dim (was 64)

    # NPE inference
    NPE_POSTERIOR_SAMPLES: int = 20000

    # NPE SBC
    RUN_SBC: bool = True
    NPE_SBC_NUM_DATASETS: int = 200
    NPE_SBC_POST_SAMPLES: int = 200


RUN_CONFIG_PARAMS = RunConfig()