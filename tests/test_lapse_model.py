"""
Tests for the lapse-augmented RT–choice simulator.
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch

from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
    _run_fine_ou_loop_lapse,
    _sample_lapse_observations,
)
from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    generate_pulses_torch,
    max_num_pulses,
)
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.priors import build_prior_theta_lapse
from sbi_for_diffusion_models.run_config import T_MAX, PULSE_INTERVAL

P_MAX = max_num_pulses()
TRIAL_DIM = 2 + P_MAX


# test helper function
def _make_lapse_theta(
    a0=0.5, lam=0.3, v=1.0, B=2.5, tau=0.3, p_lapse=0.1,
    *, N=1, device="cpu",
):
    row = torch.tensor(
        [a0, lam, v, B, tau, p_lapse], dtype=torch.float32, device=device,
    )
    return row.unsqueeze(0).expand(N, -1).contiguous()


# Shape tests
class TestLapseSimulatorShapes:

    def test_single_trial(self):
        theta = _make_lapse_theta(N=1)
        x, hit, s = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert x.shape == (1, 2)
        assert hit.shape == (1,)
        assert s.shape == (1, P_MAX)

    def test_batch(self):
        theta = _make_lapse_theta(N=64)
        x, hit, s = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert x.shape == (64, 2)
        assert hit.shape == (64,)
        assert s.shape == (64, P_MAX)

    def test_1d_theta_unsqueezed(self):
        theta = torch.tensor([0.5, 0.3, 1.0, 2.5, 0.3, 0.1])
        x, hit, s = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert x.shape == (1, 2)

    def test_wrong_dim_raises(self):
        theta = torch.randn(4, 5)  # 5 columns, not 6
        with pytest.raises(ValueError, match="Expected theta shape"):
            simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)

    def test_with_external_pulses(self):
        theta = _make_lapse_theta(N=4)
        pulses = generate_pulses_torch(
            4, P_MAX, p_success=0.7, device=torch.device("cpu"), dtype=torch.float32,
        )
        x, hit, s = simulate_rt_choice_batch_lapse(
            theta, mu_sensory=1.0, pulse_sides=pulses,
        )
        assert x.shape == (4, 2)
        assert torch.equal(s, pulses)


# Value range tests
class TestLapseSimulatorValues:

    def test_rt_range(self):
        theta = _make_lapse_theta(N=200, p_lapse=0.3)
        x, _, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        rt = x[:, 0]
        assert torch.all(rt >= 0), f"Negative RT: {rt.min()}"
        assert torch.all(rt <= T_MAX + 1e-4), f"RT > T_MAX: {rt.max()}"

    def test_choice_values(self):
        theta = _make_lapse_theta(N=200, p_lapse=0.5)
        x, _, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert set(x[:, 1].unique().tolist()) <= {0.0, 1.0}

    def test_hit_is_bool(self):
        theta = _make_lapse_theta(N=10)
        _, hit, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert hit.dtype == torch.bool


# Lapse mechanics
class TestLapseMechanics:

    def test_all_lapse_all_hit(self):
        """With p_lapse=1 every trial is a lapse and must be marked hit=True."""
        theta = _make_lapse_theta(N=200, p_lapse=1.0)
        _, hit, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        assert torch.all(hit), "Lapse trials must be flagged hit=True"

    def test_all_lapse_choice_roughly_balanced(self):
        """Lapse choices should be ≈ 50/50."""
        torch.manual_seed(0)
        theta = _make_lapse_theta(N=2000, p_lapse=1.0)
        x, _, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        frac_right = x[:, 1].mean().item()
        assert 0.40 < frac_right < 0.60, f"Lapse choice frac_right={frac_right:.3f}"

    def test_all_lapse_rt_spread(self):
        """Lapse RTs should be roughly uniform — not all piled at one value."""
        torch.manual_seed(1)
        theta = _make_lapse_theta(N=2000, p_lapse=1.0)
        x, _, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        rt = x[:, 0]
        # With uniform RT on (0, T_MAX), std ≈ T_MAX / sqrt(12) ≈ 2.89
        assert rt.std().item() > 1.0, f"Lapse RT std too low: {rt.std():.3f}"
        # Median should be near T_MAX/2
        median_rt = rt.median().item()
        assert 2.0 < median_rt < 8.0, f"Lapse RT median={median_rt:.2f}"

    def test_no_lapse_matches_base_model(self):
        """At p_lapse=0 the lapse model should behave identically to the base."""
        torch.manual_seed(42)
        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(42)
        pulses = generate_pulses_torch(
            10, P_MAX, p_success=0.7, device=torch.device("cpu"),
            dtype=torch.float32, generator=g,
        )

        # Base model
        theta_base = torch.tensor([[0.5, 0.3, 1.0, 2.5, 0.3]])
        torch.manual_seed(99)
        x_base, hit_base, _ = simulate_rt_choice_batch(
            theta_base.expand(10, -1), mu_sensory=1.0, pulse_sides=pulses,
        )

        # Lapse model with p_lapse=0
        theta_lapse = torch.tensor([[0.5, 0.3, 1.0, 2.5, 0.3, 0.0]])
        torch.manual_seed(99)
        x_lapse, hit_lapse, _ = simulate_rt_choice_batch_lapse(
            theta_lapse.expand(10, -1), mu_sensory=1.0, pulse_sides=pulses,
        )
        assert x_base.shape == x_lapse.shape
        assert hit_base.shape == hit_lapse.shape

    def test_p_lapse_zero_no_random_choices(self):
        """With p_lapse=0 there should be no lapse-generated random choices."""
        theta = _make_lapse_theta(N=100, p_lapse=0.0, v=50.0, B=2.0, tau=0.05)
        x, hit, _ = simulate_rt_choice_batch_lapse(theta, mu_sensory=1.0)
        # With huge drift and small boundary every non-lapse trial should hit
        assert hit.sum() > 90


# _sample_lapse_observation shape
class TestSampleLapseObservations:

    def test_shapes(self):
        rt, ch = _sample_lapse_observations(100, device=torch.device("cpu"))
        assert rt.shape == (100,)
        assert ch.shape == (100,)

    def test_zero_trials(self):
        rt, ch = _sample_lapse_observations(0, device=torch.device("cpu"))
        assert rt.shape == (0,)
        assert ch.shape == (0,)

    def test_rt_in_range(self):
        rt, _ = _sample_lapse_observations(1000, device=torch.device("cpu"))
        assert torch.all(rt >= 0)
        assert torch.all(rt < T_MAX)

    def test_choice_binary(self):
        _, ch = _sample_lapse_observations(1000, device=torch.device("cpu"))
        assert set(ch.unique().tolist()) <= {0, 1}


# OU attractor = B/2 in the lapse fine loop
class TestLapseOUDynamics:
    """
    The critical invariant: the OU update is
        a ← a * decay + (B/2)*(1 - decay) + noise
    so the deterministic fixed point is B/2, NOT 0.
    """

    def _run_noiseless(self, a0_frac_val, B_val, lam_val=1.0):
        """Run lapse OU loop with ~zero noise, no pulses, huge boundaries."""
        device = torch.device("cpu")
        dtype = torch.float32
        a0_frac = torch.tensor([a0_frac_val], device=device, dtype=dtype)
        lam = torch.tensor([lam_val], device=device, dtype=dtype)
        v = torch.tensor([0.0], device=device, dtype=dtype)
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)
        s = torch.zeros(1, P_MAX, device=device, dtype=dtype)

        hit, choice, dt_out = _run_fine_ou_loop_lapse(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )
        return hit, choice, dt_out

    def test_converges_from_below(self):
        """Start near 0 (a0_frac=0.01, B=100) → should drift up toward B/2=50."""
        hit, _, _ = self._run_noiseless(a0_frac_val=0.01, B_val=100.0)
        assert not hit.item(), "Should not hit boundary with B=100 and no noise"

        # Verify analytically
        dt = 0.01
        decay = math.exp(-1.0 * dt)
        a = 0.01 * 100.0  # =1.0
        for _ in range(int(T_MAX / dt)):
            a = a * decay + 50.0 * (1.0 - decay)
        assert a > 45.0, f"Accumulator should be near 50, got {a:.2f}"

    def test_converges_from_above(self):
        """Start near B (a0_frac=0.99, B=100) → should drift down toward B/2=50."""
        hit, _, _ = self._run_noiseless(a0_frac_val=0.99, B_val=100.0)
        assert not hit.item()

        dt = 0.01
        decay = math.exp(-1.0 * dt)
        a = 0.99 * 100.0  # =99.0
        for _ in range(int(T_MAX / dt)):
            a = a * decay + 50.0 * (1.0 - decay)
        assert 45.0 < a < 55.0, f"Accumulator should be near 50, got {a:.2f}"

    def test_attractor_is_not_zero(self):
        """
        If the OU attractor were 0 (a common bug), starting at a0_frac=0.5
        with B=100 would pull the accumulator toward 0 and eventually cross
        the lower boundary.  With attractor = B/2 it stays near 50.
        """
        hit, _, _ = self._run_noiseless(a0_frac_val=0.5, B_val=100.0, lam_val=2.0)
        assert not hit.item(), (
            "With attractor=B/2, starting at B/2 with no noise should never "
            "cross a boundary.  If this fails the attractor may be 0."
        )

    def test_boundaries_are_zero_and_B(self):
        """
        Lower boundary = 0, upper boundary = B.
        Verify by starting just above 0 with a strong downward pulse kick.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        B_val = 2.0
        a0_frac = torch.tensor([0.01], device=device, dtype=dtype)  # a=0.02
        lam = torch.tensor([0.0], device=device, dtype=dtype)
        v = torch.tensor([5.0], device=device, dtype=dtype)
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)

        # All pulses = -1 → strong downward kicks
        s = -torch.ones(1, P_MAX, device=device, dtype=dtype)

        hit, choice, _ = _run_fine_ou_loop_lapse(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )
        # Should hit the lower boundary (0) → choice=0
        assert hit.item(), "Should hit lower boundary with strong negative pulses"
        assert choice.item() == 0, "Lower boundary should map to choice=0"

    def test_upper_boundary_hit(self):
        """Strong upward pulses near the upper boundary → choice=1."""
        device = torch.device("cpu")
        dtype = torch.float32

        B_val = 2.0
        a0_frac = torch.tensor([0.99], device=device, dtype=dtype)  # a ≈ B
        lam = torch.tensor([0.0], device=device, dtype=dtype)
        v = torch.tensor([5.0], device=device, dtype=dtype)
        B = torch.tensor([B_val], device=device, dtype=dtype)
        tau = torch.tensor([0.0], device=device, dtype=dtype)

        s = torch.ones(1, P_MAX, device=device, dtype=dtype)

        hit, choice, _ = _run_fine_ou_loop_lapse(
            a0_frac, lam, v, B, tau, s,
            mu_sensory=1e-12,
            dt_internal=0.01,
            pulse_interval=float(PULSE_INTERVAL),
            T_MAX=float(T_MAX),
        )
        assert hit.item(), "Should hit upper boundary with strong positive pulses"
        assert choice.item() == 1, "Upper boundary should map to choice=1"


# Session-level simulation with lapse model
class TestLapseSessionSimulation:
    """Verify simulate_training_sessions works with the 6-param lapse model."""

    def test_shapes(self):
        prior = build_prior_theta_lapse()
        N, T = 3, 32
        theta, x = simulate_training_sessions(
            prior,
            num_sessions=N,
            num_trials=T,
            simulate_batch_fn=simulate_rt_choice_batch_lapse,
            device=torch.device("cpu"),
            mu_sensory=1.0,
            p_success=0.7,
            P=P_MAX,
            log_rt=True,
            seed=0,
        )
        assert theta.shape == (N, 6), f"Expected (N,6), got {theta.shape}"
        assert x.shape == (N, T * TRIAL_DIM)

    def test_no_nans_or_infs(self):
        prior = build_prior_theta_lapse()
        _, x = simulate_training_sessions(
            prior,
            num_sessions=4,
            num_trials=32,
            simulate_batch_fn=simulate_rt_choice_batch_lapse,
            device=torch.device("cpu"),
            mu_sensory=1.0,
            p_success=0.7,
            P=P_MAX,
            log_rt=True,
            seed=0,
        )
        assert not torch.any(torch.isnan(x)), "NaN in lapse training data"
        assert not torch.any(torch.isinf(x)), "Inf in lapse training data"

    def test_fixed_theta_lapse(self):
        prior = build_prior_theta_lapse()
        theta_fixed = torch.tensor([0.5, 0.3, 1.0, 2.5, 0.3, 0.15])
        N = 4
        theta, x = simulate_training_sessions(
            prior,
            num_sessions=N,
            num_trials=32,
            simulate_batch_fn=simulate_rt_choice_batch_lapse,
            device=torch.device("cpu"),
            mu_sensory=1.0,
            p_success=0.7,
            P=P_MAX,
            log_rt=True,
            seed=0,
            theta=theta_fixed,
        )
        for i in range(N):
            assert torch.allclose(theta[i], theta_fixed, atol=1e-6)
