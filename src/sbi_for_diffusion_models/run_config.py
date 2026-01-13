from __future__ import annotations
from dataclasses import dataclass 

@dataclass(frozen=True)
class RunConfig:
    # Data / simulator settings
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.75

    # Training settings
    NUM_SIMULATIONS: int = 10_000
    TRAIN_BATCH_SIZE: int = 4096

    # Start small; likelihood approximation bias can explode when summing over many trials.
    NUM_TRIALS_OBS : int = 50

    # We recommend log-transforming RT but NOT the categorical choice.
    LOG_RT_MANUALLY: bool = False

    """
    If your sbi version supports log_transform_x for MNLE (log RT but not choice),
    you can set LOG_RT_MANUALLY=False and SBI_LOG_TRANSFORM_X=True
    """
    SBI_LOG_TRANSFORM_X: bool = True
    Z_SCORE_X: str | None = "independent"

    # MCMC settings
    NUM_CHAINS: int = 2
    WARMUP_STEPS: int = 100
    POSTERIOR_SAMPLES: int = 1000

    """
    Optional likelihood tempering for debugging only (1.0 = true posterior).
    If you see crazy posteriors at large NUM_TRIALS_OBS, try TEMPERATURE=10 or 100 to diagnose.
    """
    TEMPERATURE: float = 1.0
    THETA_TRUE_FROM_PRIOR: bool = True

    # SBC settings 
    SBC_NUM_DATASETS: int = 10 
    SBC_POST_SAMPLES: int = 1500


RUN_CONFIG_PARAMS = RunConfig()