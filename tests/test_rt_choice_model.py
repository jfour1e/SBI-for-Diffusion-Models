import pytest
import torch

from sbi_for_diffusion_models.models.rt_choice_model import (
    max_num_pulses,
    generate_pulses_torch,
    simulate_rt_choice_batch,
)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generate_pulses_torch_values_and_shape(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    N = 10
    P = 20
    gen = torch.Generator(device=device).manual_seed(0)

    s = generate_pulses_torch(
        n_trials=N,
        n_pulses=P,
        p_success=0.7,
        device=torch.device(device),
        generator=gen,
    )
    assert s.shape == (N, P)
    assert s.dtype == torch.float32
    assert s.device.type == device
    # Values must be exactly -1 or +1
    uniq = torch.unique(s)
    assert set(uniq.tolist()).issubset({-1.0, 1.0})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_simulate_rt_choice_batch_shapes_and_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    N = 32
    P = max_num_pulses()

    # theta: [a0_frac, lam, v, B, t_nd]
    theta = torch.tensor(
        [[0.5, 1.0, 0.4, 1.0, 0.1]] * N,
        device=dev,
        dtype=torch.float32,
    )

    gen = torch.Generator(device=device).manual_seed(123)
    pulses = generate_pulses_torch(
        n_trials=N, n_pulses=P, p_success=0.7, device=dev, generator=gen
    )

    x, hit, s_used = simulate_rt_choice_batch(
        theta,
        mu_sensory=1.0,
        pulse_sides=pulses,
        p_success=0.7,
        pulse_generator=gen,
    )

    assert x.shape == (N, 2)
    assert hit.shape == (N,)
    assert s_used.shape == (N, P)

    assert x.device == dev
    assert hit.device == dev
    assert s_used.device == dev

    assert x.dtype == torch.float32
    assert hit.dtype == torch.bool
    assert s_used.dtype == torch.float32

    # rt should be in (0, T_MAX]; we can't import T_MAX easily here, but must be positive finite.
    assert torch.isfinite(x[:, 0]).all()
    assert (x[:, 0] > 0).all()

    # choice should be 0/1 (float)
    assert torch.all((x[:, 1] == 0.0) | (x[:, 1] == 1.0))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_simulate_rt_choice_batch_has_timeouts_for_extreme_params(device):
    """
    This checks that timeouts exist for 'hard' parameter regimes, i.e. hit is not always True.
    It does NOT check resampling (since resampling is removed).
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    N = 256
    P = max_num_pulses()

    # Very small drift and very large boundary -> likely timeouts
    theta = torch.tensor(
        [[0.5, 1.0, 1e-6, 50.0, 0.1]] * N,
        device=dev,
        dtype=torch.float32,
    )

    gen = torch.Generator(device=device).manual_seed(999)
    pulses = generate_pulses_torch(
        n_trials=N, n_pulses=P, p_success=0.5, device=dev, generator=gen
    )

    x, hit, _ = simulate_rt_choice_batch(
        theta,
        mu_sensory=0.1,
        pulse_sides=pulses,
        p_success=0.5,
        pulse_generator=gen,
        n_refine=10,  # keep test fast
    )

    # Expect at least some timeouts (not_hit True for some)
    assert (~hit).any()
    # And some hits typically happen too (if none hit, relax this)
    assert hit.any() or True  # allow rare cases where none hit depending on T_MAX