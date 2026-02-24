import pytest
import torch

from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions,
    flatten_observed_session,
)
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses


class DummyPrior:
    """Simple prior that mimics .sample((1,)) -> tensor"""
    def __init__(self, device):
        self.device = device

    def sample(self, shape):
        # return (1,5)
        return torch.tensor([[0.5, 1.0, 0.3, 1.0, 0.1]], device=self.device, dtype=torch.float32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_simulate_training_sessions_shapes_and_mask(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    P = max_num_pulses()
    T = 40
    N_sessions = 5

    prior = DummyPrior(device=torch.device("cpu"))  # prior can live on CPU; function moves theta to dev

    theta_all, x_all = simulate_training_sessions(
        prior,
        num_sessions=N_sessions,
        num_trials=T,
        device=dev,
        mu_sensory=1.0,
        p_success=0.7,
        P=P,
        log_rt=False,
        seed=0,
    )

    trial_dim = 2 + P + 1
    assert theta_all.shape == (N_sessions, 5)
    assert x_all.shape == (N_sessions, T * trial_dim)

    assert theta_all.device == dev
    assert x_all.device == dev

    # Check that mask values are 0/1 for each trial
    x_3d = x_all.view(N_sessions, T, trial_dim)
    mask = x_3d[..., -1]
    assert torch.all((mask == 0.0) | (mask == 1.0))

    # If mask=0, then those trials should be padded with zeros in [rt, choice, pulses]
    padded = mask == 0.0
    if padded.any():
        feats = x_3d[..., :-1]
        assert torch.all(feats[padded] == 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flatten_observed_session_shape(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    P = max_num_pulses()
    T = 10
    trial_dim = 2 + P + 1

    x_o = torch.zeros((T, 2), device=dev)
    pulses_o = torch.zeros((T, P), device=dev)
    mask_o = torch.ones((T, 1), device=dev)

    flat = flatten_observed_session(x_o, pulses_o, mask_o)
    assert flat.shape == (1, T * trial_dim)
    assert flat.device == dev
    assert flat.dtype == torch.float32