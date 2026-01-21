from __future__ import annotations

import math
import numpy as np
import pytest
import torch
from torch import Tensor

from torch.distributions import Beta, LogNormal, Distribution
from sbi.utils import MultipleIndependent

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# package imports 
from sbi_for_diffusion_models.models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    pulse_schedule,
    n_pulses_max_from_schedule,
    generate_pulse_matrix_numpy,
    generate_pulse_sides,  
)
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_set_with_conditions,
    simulate_observed_session,
)
from sbi_for_diffusion_models.proposals import (
    PulseSequenceProposal,
    ExtendedProposal,
)
from sbi_for_diffusion_models.mnle import run_inference_mcmc
from sbi_for_diffusion_models.potentials import ConditionedMNLELogLikelihood

# Helper functions

def _make_prior_theta() -> Distribution:
    """
    Prior over theta = [a0, lam, v, B, tau].
    """
    return MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),          # a0
            LogNormal(torch.tensor([-1.0]), torch.tensor([1.0])),     # lam
            LogNormal(torch.tensor([0.0]), torch.tensor([1.0])),      # v
            LogNormal(torch.tensor([2.75]), torch.tensor([0.5])),     # B
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),           # tau (placeholder Beta)
        ]
    )

class _CfgTest:
    """
    Only includes attributes actually used by the functions under test.
    """
    MU_SENSORY = 1.0
    P_SUCCESS = 0.75

    # training sim
    NUM_SIMULATIONS = 2048
    TRAIN_BATCH_SIZE = 512
    LOG_RT_MANUALLY = False

    # observed sim
    NUM_TRIALS_OBS = 25

    # MCMC
    NUM_CHAINS = 1
    WARMUP_STEPS = 25
    POSTERIOR_SAMPLES = 50
    TEMPERATURE = 1.0

class DummyLikelihoodEstimator(torch.nn.Module):
    """
    Differentiable dummy estimator that mimics the API used inside
    ConditionedMNLELogLikelihood:

        estimator.log_prob(x, condition=theta_with_condition)

    Returns (batch,) log-prob values.
    """
    def log_prob(self, x: Tensor, condition: Tensor) -> Tensor:
        # x: (B,2) or (1,B,2) depending on how it's passed
        # Condition: (B, 5+P)
        if x.ndim == 3:
            # your ConditionedMNLELogLikelihood uses x_repeated shape (1, B, 2)
            x = x.squeeze(0)

        # Make a simple synthetic "mean" for rt from condition.
        # Use a smooth function of (theta,pulses) so gradients exist.
        # Keep it stable: mean_rt in ~[0.1, 2.0].
        theta = condition[:, :5]
        pulses = condition[:, 5:]

        mean_rt = 0.5 + 0.05 * torch.tanh(theta[:, 2]) + 0.02 * pulses.mean(dim=1)
        rt = x[:, 0]
        choice = x[:, 1]

        # Gaussian-ish rt term + small choice penalty (treat as float)
        rt_ll = -0.5 * ((rt - mean_rt) / 0.3) ** 2
        choice_ll = -0.05 * (choice - 1.0) ** 2
        return rt_ll + choice_ll


def _compute_manual_ll_sum(estimator: DummyLikelihoodEstimator, theta: Tensor, x_o: Tensor, pulses_o: Tensor) -> Tensor:
    """
    Manual: sum_i log p(x_i | theta, pulse_i).
    Returns (N,) where N = number of thetas.
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    N = theta.shape[0]
    T = x_o.shape[0]
    P = pulses_o.shape[1]

    # Build batch of (N*T) pairs
    theta_rep = theta.repeat_interleave(T, dim=0)          # (N*T,5)
    pulses_rep = pulses_o.repeat(N, 1)                      # (N*T,P)
    cond = torch.cat([theta_rep, pulses_rep], dim=1)        # (N*T, 5+P)

    x_rep = x_o.repeat(N, 1)                                # (N*T,2)
    ll = estimator.log_prob(x_rep, condition=cond)          # (N*T,)
    ll = ll.view(N, T).sum(dim=1)                           # (N,)
    return ll


"""
Simulator Tests 
"""
def test_simulator_shape_dtype_and_choice_domain():
    torch.manual_seed(0)
    N = 128
    theta = torch.randn(N, 5, dtype=torch.float32)
    x = rt_choice_model_simulator_torch(theta, mu_sensory=1.0, pulse_sides=None, p_success=0.75)

    assert isinstance(x, torch.Tensor)
    assert x.shape == (N, 2)
    assert x.dtype == torch.float32

    # choice must be 0/1/2 (stored as float)
    choices = x[:, 1].to(torch.int64)
    assert torch.all((choices == 0) | (choices == 1) | (choices == 2))

def test_simulator_outputs_finite_and_rt_in_range():
    torch.manual_seed(1)
    N = 256
    theta = torch.randn(N, 5, dtype=torch.float32)
    x = rt_choice_model_simulator_torch(theta, mu_sensory=1.0, pulse_sides=None, p_success=0.75)

    assert torch.isfinite(x).all()
    rt = x[:, 0]
    # strictly positive and bounded by T_MAX (your simulator clamps)
    assert torch.all(rt > 0.0)
    assert torch.all(rt <= 8.0 + 1e-6)  # T_MAX in your constants is 8.0

def test_simulator_edge_case_theta_vector_and_broadcast_pulses():
    torch.manual_seed(2)
    # theta as (5,)
    theta = torch.randn(5, dtype=torch.float32)

    # derive P from schedule
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

    # pulses as (P,) should broadcast
    pulses = torch.ones(P, dtype=torch.float32)
    x = rt_choice_model_simulator_torch(theta, mu_sensory=1.0, pulse_sides=pulses, p_success=0.75)

    assert x.shape == (1, 2)
    assert torch.isfinite(x).all()
    assert int(x[0, 1].item()) in (0, 1, 2)

"""
Pulse Generation Tests 
"""
def test_generate_pulse_sides_domain():
    rng = np.random.default_rng(0)
    P = 50
    s = generate_pulse_sides(rng, P, p_success=0.75)
    assert s.shape == (P,)
    assert s.dtype == np.float32
    assert np.all(np.isin(s, [-1.0, 1.0]))

def test_generate_pulse_matrix_numpy_shape_values():
    rng = np.random.default_rng(1)
    n_trials, n_pulses = 10, 40
    S = generate_pulse_matrix_numpy(rng, n_trials=n_trials, n_pulses=n_pulses, p_success=0.75)
    assert S.shape == (n_trials, n_pulses)
    assert S.dtype == np.float32
    assert np.all(np.isin(S, [-1.0, 1.0]))

"""
Data Pipeline Test 
"""
# def test_simulate_training_set_with_conditions_shapes_and_values():
#     cfg = _CfgTest()
#     torch.manual_seed(0)
#     np.random.seed(0)

#     n_max, steps_per_pulse = pulse_schedule()
#     P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

#     prior_theta = _make_prior_theta()
#     pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0, device="cpu")
#     proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

#     z_train, x_train = simulate_training_set_with_conditions(
#         proposal=proposal_z,
#         num_simulations=cfg.NUM_SIMULATIONS,
#         batch_size=cfg.TRAIN_BATCH_SIZE,
#         device="cpu",
#         mu_sensory=cfg.MU_SENSORY,
#         p_success=cfg.P_SUCCESS,
#         P=P,
#         log_rt=cfg.LOG_RT_MANUALLY,
#     )

#     assert z_train.shape == (cfg.NUM_SIMULATIONS, 5 + P)
#     assert x_train.shape == (cfg.NUM_SIMULATIONS, 2)
#     assert torch.isfinite(z_train).all()
#     assert torch.isfinite(x_train).all()

#     # choice domain
#     choices = x_train[:, 1].to(torch.int64)
#     assert torch.all((choices == 0) | (choices == 1) | (choices == 2))

# def test_simulate_observed_session_shapes_and_consistency():
#     cfg = _CfgTest()
#     torch.manual_seed(0)
#     np.random.seed(0)

#     n_max, steps_per_pulse = pulse_schedule()
#     P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

#     prior_theta = _make_prior_theta()
#     theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)

#     x_o, pulses_o = simulate_observed_session(
#         theta_true,
#         num_trials=cfg.NUM_TRIALS_OBS,
#         device="cpu",
#         mu_sensory=cfg.MU_SENSORY,
#         p_success=cfg.P_SUCCESS,
#         P=P,
#         seed=123,
#         log_rt=cfg.LOG_RT_MANUALLY,
#     )

#     assert x_o.shape == (cfg.NUM_TRIALS_OBS, 2)
#     assert pulses_o.shape == (cfg.NUM_TRIALS_OBS, P)
#     assert torch.isfinite(x_o).all()
#     assert torch.isfinite(pulses_o).all()

"""
Conditioned MNLE Tests
"""
def test_conditioned_mnle_loglike_matches_manual_sum():
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = _CfgTest()
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

    # Simulate one observed dataset
    prior_theta = _make_prior_theta()
    theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=cfg.NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=cfg.LOG_RT_MANUALLY,
    )

    # Create dummy estimator + conditioned loglike module
    estimator = DummyLikelihoodEstimator()
    conditioned = ConditionedMNLELogLikelihood(estimator=estimator, local_theta=pulses_o, device="cpu")

    # Evaluate at a batch of thetas
    thetas = prior_theta.sample((7,)).to(torch.float32)
    ll_fast = conditioned(thetas, x_o, track_gradients=False)  # (7,)

    ll_manual = _compute_manual_ll_sum(estimator, thetas, x_o, pulses_o)  # (7,)

    assert ll_fast.shape == (7,)
    assert torch.isfinite(ll_fast).all()
    assert torch.allclose(ll_fast, ll_manual, atol=1e-5, rtol=1e-5)

"""
Inference Tests
"""
# def test_run_inference_mcmc_outputs_finite_samples():
#     torch.manual_seed(0)
#     np.random.seed(0)

#     cfg = _CfgTest()
#     n_max, steps_per_pulse = pulse_schedule()
#     P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

#     prior_theta = _make_prior_theta()
#     theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)

#     x_o, pulses_o = simulate_observed_session(
#         theta_true,
#         num_trials=cfg.NUM_TRIALS_OBS,
#         device="cpu",
#         mu_sensory=cfg.MU_SENSORY,
#         p_success=cfg.P_SUCCESS,
#         P=P,
#         seed=123,
#         log_rt=cfg.LOG_RT_MANUALLY,
#     )

#     # Use dummy estimator so MCMC is cheap and does not depend on training MNLE.
#     estimator = DummyLikelihoodEstimator()

#     samples = run_inference_mcmc(cfg, prior_theta, estimator, x_o, pulses_o)

#     assert isinstance(samples, torch.Tensor)
#     assert samples.shape == (cfg.POSTERIOR_SAMPLES, 5)
#     assert samples.device.type == "cpu"
#     assert torch.isfinite(samples).all()
