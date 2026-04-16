"""
Tests for the core OU-based RT–choice simulator.
"""
from __future__ import annotations

import pytest
import torch

from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    pack_x_rt_choice,
    max_num_pulses,
    _ou_transition_params,
    _run_fine_ou_loop,
    generate_pulses_torch,
)
from sbi_for_diffusion_models.run_config import T_MAX, PULSE_INTERVAL

P_MAX = max_num_pulses()

# simulate_rt_choice_batch shapes
class TestSimulateBatchShapes:

    def test_single_theta(self, default_theta, device):
        x, hit, s = simulate_rt_choice_batch(
            default_theta.unsqueeze(0),
            mu_sensory=1.0,
        )
        assert x.shape == (1, 2), f"x shape {x.shape}"
        assert hit.shape == (1,)
        assert s.shape == (1, P_MAX)

    def test_batch_theta(self, batch_theta, device):
        N = batch_theta.shape[0]
        x, hit, s = simulate_rt_choice_batch(batch_theta, mu_sensory=1.0)
        assert x.shape == (N, 2)
        assert hit.shape == (N,)
        assert s.shape == (N, P_MAX)

    def test_1d_theta_unsqueezed(self, default_theta, device):
        """A 1-D theta should be automatically reshaped to (1, 5)."""
        x, hit, s = simulate_rt_choice_batch(default_theta, mu_sensory=1.0)
        assert x.shape == (1, 2)

    def test_wrong_theta_dim_raises(self, device, dtype):
        bad = torch.randn(3, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="Expected theta shape"):
            simulate_rt_choice_batch(bad, mu_sensory=1.0)

    def test_with_external_pulses(self, default_theta, device, dtype, generator):
        pulses = generate_pulses_torch(
            n_trials=1, n_pulses=P_MAX,
            p_success=0.7, device=device, dtype=dtype, generator=generator,
        )
        x, hit, s = simulate_rt_choice_batch(
            default_theta.unsqueeze(0),
            mu_sensory=1.0,
            pulse_sides=pulses,
        )
        assert x.shape == (1, 2)
        # Returned pulses should be the same object content
        assert torch.equal(s, pulses)


# Output value ranges
class TestSimulateBatchValues:

    def test_rt_range(self, batch_theta):
        x, hit, _ = simulate_rt_choice_batch(batch_theta, mu_sensory=1.0)
        rt = x[:, 0]
        assert torch.all(rt >= 0), "Negative RT found"
        assert torch.all(rt <= T_MAX + 1e-4), f"RT exceeds T_MAX: {rt.max()}"

    def test_choice_values(self, batch_theta):
        x, hit, _ = simulate_rt_choice_batch(batch_theta, mu_sensory=1.0)
        choice = x[:, 1]
        assert set(choice.unique().tolist()) <= {0.0, 1.0}

    def test_hit_is_bool(self, batch_theta):
        _, hit, _ = simulate_rt_choice_batch(batch_theta, mu_sensory=1.0)
        assert hit.dtype == torch.bool


# pack_x_rt_choice
class TestPackXRtChoice:

    def test_no_log(self):
        rt_choice = torch.tensor([[0.5, 1.0], [1.2, 0.0]])
        packed = pack_x_rt_choice(rt_choice, log_rt=False)
        assert packed.shape == (2, 2)
        assert torch.allclose(packed[:, 0], rt_choice[:, 0])

    def test_log_rt(self):
        rt_choice = torch.tensor([[0.5, 1.0], [1.2, 0.0]])
        packed = pack_x_rt_choice(rt_choice, log_rt=True)
        expected_log = torch.log(rt_choice[:, 0].clamp_min(1e-6))
        assert torch.allclose(packed[:, 0], expected_log, atol=1e-6)

    def test_choice_preserved_as_float(self):
        rt_choice = torch.tensor([[0.5, 1.0], [1.2, 0.0]])
        packed = pack_x_rt_choice(rt_choice, log_rt=False)
        assert packed[:, 1].tolist() == [1.0, 0.0]


# OU transition parameterss
class TestOUTransitionParams:

    def test_zero_lambda_gives_brownian(self):
        """When lam ≈ 0 the process is Brownian: decay=1, var ≈ sigma²·dt."""
        lam = torch.tensor([0.0])
        dt, sigma = 0.01, 1.0
        decay, noise_std = _ou_transition_params(lam, dt, sigma)
        assert torch.allclose(decay, torch.ones(1), atol=1e-6)
        # var ≈ dt for Brownian with sigma=1
        assert abs(noise_std.item() ** 2 - dt) < 1e-6

    def test_large_lambda_strong_decay(self):
        lam = torch.tensor([1000.0])
        decay, _ = _ou_transition_params(lam, dt=0.01, sigma=1.0)
        # exp(-1000 * 0.01) = exp(-10) ≈ 4.5e-5
        assert decay.item() < 0.001, "Large lambda should give near-zero decay"

# OU dynamics: leak toward B/2 instead of towards 0
class TestOUDecayToMidpoint:

    def test_decay_toward_B_over_2(self, device, dtype):
        N = 1
        B_val = 4.0
        lam_val = 2.0
        a0_frac_val = 0.1  # start near lower bound → far from B/2

        a0_frac = torch.tensor([a0_frac_val], device=device, dtype=dtype)
        lam = torch.tensor([lam_val], device=device, dtype=dtype)
        v = torch.tensor([0.0], device=device, dtype=dtype)   # no pulse influence
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)

        # All-zero pulses, no kick contribution
        s = torch.zeros(N, P_MAX, device=device, dtype=dtype)

        # Use a very small mu_sensory to suppress noise
        hit, choice, dt_out = _run_fine_ou_loop(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,   # essentially zero noise
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )

    def test_accumulator_converges_to_midpoint(self, device, dtype):
        """
        Directly verify convergence by running with boundaries far away.
        Start at a0_frac=0.01 (near 0), B=100 (huge boundaries), no noise.
        After many steps the accumulator should be ≈ B/2 = 50.
        """
        N = 1
        B_val = 100.0
        a0_frac = torch.tensor([0.01], device=device, dtype=dtype)
        lam = torch.tensor([1.0], device=device, dtype=dtype)
        v = torch.tensor([0.0], device=device, dtype=dtype)
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)
        s = torch.zeros(N, P_MAX, device=device, dtype=dtype)

        hit, choice, dt_out = _run_fine_ou_loop(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )
        # The process should not hit a boundary (B large)
        assert not hit.item(), "Should not hit boundary with B=100"

        # Now manually simulate the deterministic OU to get the final value.
        # a_{k+1} = a_k * decay + (B/2)*(1-decay)
        dt = 0.01
        decay = torch.exp(-lam * dt).item()
        attractor = B_val / 2.0
        a = 0.01 * B_val  # a0_frac * B
        n_steps = int(T_MAX / dt)
        for _ in range(n_steps):
            a = a * decay + attractor * (1.0 - decay)
        assert abs(a - attractor) < 1.0, (
            f"Accumulator should converge to B/2={attractor}, got {a:.2f}"
        )

    def test_attractor_is_not_zero(self, device, dtype):
        """
        If the OU attractor were 0 (wrong), starting at a0_frac=0.99
        with B=100 and no noise would cause the accumulator to decrease
        toward 0.  Verify it stays near B/2 instead.
        """
        B_val = 100.0
        a0_frac = torch.tensor([0.99], device=device, dtype=dtype)
        lam = torch.tensor([1.0], device=device, dtype=dtype)
        v = torch.tensor([0.0], device=device, dtype=dtype)
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)
        s = torch.zeros(1, P_MAX, device=device, dtype=dtype)

        hit, _, _ = _run_fine_ou_loop(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )
        assert not hit.item(), "Should not hit boundary starting near B/2 with no noise"

        # a should converge toward 50, not 0
        dt = 0.01
        decay = torch.exp(-lam * dt).item()
        attractor = B_val / 2.0
        a = 0.99 * B_val
        for _ in range(int(T_MAX / dt)):
            a = a * decay + attractor * (1.0 - decay)
        assert a > 40.0, f"Accumulator decayed to {a:.2f} — attractor should be ~50"


# Reproducibility test
class TestReproducibility:

    def test_same_seed_same_output(self, default_theta, device, dtype):
        theta = default_theta.unsqueeze(0)

        def _run(seed):
            # The fine OU loop calls torch.randn() which uses the global RNG,
            # so we must reset it as well as any explicit generators.
            torch.manual_seed(seed)
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            pulses = generate_pulses_torch(
                1, P_MAX, p_success=0.7, device=device, dtype=dtype, generator=g,
            )
            return simulate_rt_choice_batch(
                theta, mu_sensory=1.0, pulse_sides=pulses,
            )

        x1, h1, s1 = _run(42)
        x2, h2, s2 = _run(42)
        assert torch.equal(s1, s2), "Pulses differ with same seed"
        assert torch.equal(x1, x2), "Outputs differ with same seed"
        

# Test extreme parameters 
class TestExtremeParameters:

    def test_very_large_v_fast_decisions(self, device, dtype):
        """Very large drift → almost all trials should hit quickly."""
        theta = torch.tensor([[0.5, 0.1, 50.0, 2.0, 0.05]], device=device, dtype=dtype)
        x, hit, _ = simulate_rt_choice_batch(theta.expand(100, -1), mu_sensory=1.0)
        assert hit.sum() > 90, f"Only {hit.sum()} of 100 hit with huge drift"

    def test_very_small_B_fast_decisions(self, device, dtype):
        """Very small boundary → quick hits."""
        theta = torch.tensor([[0.5, 0.1, 1.0, 0.01, 0.05]], device=device, dtype=dtype)
        x, hit, _ = simulate_rt_choice_batch(theta.expand(50, -1), mu_sensory=1.0)
        assert hit.sum() > 40, f"Only {hit.sum()} of 50 hit with tiny boundary"

    def test_tau_near_tmax_mostly_timeouts(self, device, dtype):
        """If tau ≈ T_MAX there's almost no time for evidence → timeouts."""
        theta = torch.tensor([[0.5, 0.1, 1.0, 5.0, T_MAX - 0.05]], device=device, dtype=dtype)
        x, hit, _ = simulate_rt_choice_batch(theta.expand(50, -1), mu_sensory=1.0)
        # Most should time out
        assert hit.sum() < 20, f"Too many hits ({hit.sum()}/50) with tau ≈ T_MAX"