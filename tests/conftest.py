from __future__ import annotations

import pytest
import torch
import numpy as np

T_MAX = 10.0
PULSE_INTERVAL = 0.25
P_MAX = int(T_MAX / PULSE_INTERVAL)
DT_INTERNAL = 0.01
MU_SENSORY = 1.0
P_SUCCESS = 0.7
NUM_TRIALS = 256
TRIAL_DIM = 2 + P_MAX


@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def dtype():
    return torch.float32

@pytest.fixture
def generator(device):
    g = torch.Generator(device=device)
    g.manual_seed(42)
    return g

@pytest.fixture
def default_theta(device, dtype):
    """A single well-behaved theta = [a0, lam, v, B, tau]."""
    return torch.tensor([0.5, 0.3, 1.0, 2.5, 0.3], device=device, dtype=dtype)

@pytest.fixture
def batch_theta(device, dtype):
    """A small batch of 8 theta vectors."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=(8, 5)).astype(np.float32)
    # Clip into reasonable ranges so simulations don't time out
    raw[:, 0] = np.clip(raw[:, 0], 0.1, 0.9)   # a0
    raw[:, 1] = np.clip(raw[:, 1], 0.05, 0.8)   # lam
    raw[:, 2] = np.clip(raw[:, 2], 0.5, 2.0)    # v
    raw[:, 3] = np.clip(raw[:, 3], 1.0, 4.0)    # B
    raw[:, 4] = np.clip(raw[:, 4], 0.1, 1.5)    # tau
    return torch.from_numpy(raw).to(device=device, dtype=dtype)