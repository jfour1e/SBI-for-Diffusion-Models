from __future__ import annotations
from dataclasses import dataclass 

@dataclass(frozen=True)
class RunConfig:
    # Data / simulator settings
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.75

    # Training settings
    NUM_SIMULATIONS: int = 100_000
    TRAIN_BATCH_SIZE: int = 4096

    # Start small
    NUM_TRIALS_OBS : int = 1000

    # We recommend log-transforming RT but NOT the categorical choice.
    LOG_RT_MANUALLY: bool = False

    """
    If your sbi version supports log_transform_x for MNLE (log RT but not choice),
    you can set LOG_RT_MANUALLY=False and SBI_LOG_TRANSFORM_X=True
    """
    SBI_LOG_TRANSFORM_X: bool = True
    Z_SCORE_X: str | None = "independent"

    # MCMC settings
    MCMC_METHOD: str = "slice_np_vectorized"
    NUM_CHAINS: int = 12
    WARMUP_STEPS: int = 200
    POSTERIOR_SAMPLES: int = 5000

    """
    Optional likelihood tempering for debugging only (1.0 = true posterior).
    If you see crazy posteriors at large NUM_TRIALS_OBS, try TEMPERATURE=10 or 100 to diagnose.
    """
    TEMPERATURE: float = 1.0
    THETA_TRUE_FROM_PRIOR: bool = True

    # SBC settings
    SBC_NUM_DATASETS: int = 10
    SBC_POST_SAMPLES: int = 1500

    # ── NPE (session-level posterior estimation) settings ──
    NPE_NUM_SESSIONS: int = 10_000
    NPE_TRAIN_BATCH_SIZE: int = 64
    NPE_HIDDEN_FEATURES: int = 128
    NPE_NUM_TRANSFORMS: int = 5
    NPE_NUM_BINS: int = 10

    # Embedding network (DeepSets)
    NPE_TRIAL_NET_HIDDEN: int = 128
    NPE_TRIAL_NET_LAYERS: int = 3
    NPE_TRIAL_NET_OUTPUT_DIM: int = 64
    NPE_AGG_FN: str = "mean"
    NPE_POST_AGG_HIDDEN: int = 128
    NPE_POST_AGG_LAYERS: int = 2
    NPE_EMBEDDING_OUTPUT_DIM: int = 64

    # NPE inference
    NPE_POSTERIOR_SAMPLES: int = 5000

    # NPE SBC
    NPE_SBC_NUM_DATASETS: int = 100
    NPE_SBC_POST_SAMPLES: int = 5000


RUN_CONFIG_PARAMS = RunConfig()