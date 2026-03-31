"""
Tests for session-level data simulation (simulate_training_sessions).
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    max_num_pulses,
)
from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.run_config import T_MAX, PULSE_INTERVAL

P_MAX = max_num_pulses()
TRIAL_DIM = 2 + P_MAX

# Shape tests
class TestSimulateTrainingSessionsShapes:

    @pytest.fixture(autouse=True)
    def _setup(self, device):
        self.device = device
        self.prior = build_prior_theta()
        self.N, self.T = 4, 32  # small for speed
        self.P = P_MAX

    def test_output_shapes_prior_sampled(self):
        theta, x = simulate_training_sessions(
            self.prior,
            num_sessions=self.N,
            num_trials=self.T,
            simulate_batch_fn=simulate_rt_choice_batch,
            device=self.device,
            mu_sensory=1.0,
            p_success=0.7,
            P=self.P,
            log_rt=True,
            seed=0,
        )
        assert theta.shape == (self.N, 5), f"theta shape: {theta.shape}"
        assert x.shape == (self.N, self.T * TRIAL_DIM), f"x shape: {x.shape}"

    def test_output_shapes_fixed_theta(self):
        theta_fixed = torch.tensor([0.5, 0.3, 1.0, 2.5, 0.3])
        theta, x = simulate_training_sessions(
            self.prior,
            num_sessions=self.N,
            num_trials=self.T,
            simulate_batch_fn=simulate_rt_choice_batch,
            device=self.device,
            mu_sensory=1.0,
            p_success=0.7,
            P=self.P,
            log_rt=True,
            seed=0,
            theta=theta_fixed,
        )
        assert theta.shape == (self.N, 5)
        # All theta rows should be the same
        for i in range(self.N):
            assert torch.allclose(theta[i], theta_fixed.to(theta.device), atol=1e-6)

    def test_single_session(self):
        theta, x = simulate_training_sessions(
            self.prior,
            num_sessions=1,
            num_trials=self.T,
            simulate_batch_fn=simulate_rt_choice_batch,
            device=self.device,
            mu_sensory=1.0,
            p_success=0.7,
            P=self.P,
            log_rt=True,
            seed=0,
        )
        assert theta.shape == (1, 5)
        assert x.shape == (1, self.T * TRIAL_DIM)


# Content / value-range tests
class TestSimulateTrainingSessionsValues:

    @pytest.fixture(autouse=True)
    def _setup(self, device):
        self.device = device
        self.prior = build_prior_theta()
        self.N, self.T = 3, 64
        self.P = P_MAX

    def _simulate(self, log_rt: bool, seed: int = 0):
        return simulate_training_sessions(
            self.prior,
            num_sessions=self.N,
            num_trials=self.T,
            simulate_batch_fn=simulate_rt_choice_batch,
            device=self.device,
            mu_sensory=1.0,
            p_success=0.7,
            P=self.P,
            log_rt=log_rt,
            seed=seed,
        )

    def test_log_rt_values(self):
        _, x = self._simulate(log_rt=True)
        x_3d = x.view(self.N, self.T, TRIAL_DIM)
        rt_col = x_3d[:, :, 0]
        # log(1e-6) ≈ -13.8, log(T_MAX) ≈ 2.3
        assert torch.all(rt_col >= math.log(1e-6) - 0.1)
        assert torch.all(rt_col <= math.log(T_MAX) + 0.1)

    def test_raw_rt_values(self):
        _, x = self._simulate(log_rt=False)
        x_3d = x.view(self.N, self.T, TRIAL_DIM)
        rt_col = x_3d[:, :, 0]
        assert torch.all(rt_col >= 0)
        assert torch.all(rt_col <= T_MAX + 1e-4)

    def test_choice_values(self):
        _, x = self._simulate(log_rt=True)
        x_3d = x.view(self.N, self.T, TRIAL_DIM)
        choice_col = x_3d[:, :, 1]
        unique_choices = set(choice_col.unique().tolist())
        assert unique_choices <= {0.0, 1.0}, f"Unexpected choices: {unique_choices}"

    def test_pulse_values(self):
        """After masking, pulses should be in {-1, 0, +1}."""
        _, x = self._simulate(log_rt=True)
        x_3d = x.view(self.N, self.T, TRIAL_DIM)
        pulse_cols = x_3d[:, :, 2:]
        unique_pulses = set(pulse_cols.unique().tolist())
        assert unique_pulses <= {-1.0, 0.0, 1.0}, f"Unexpected pulse values: {unique_pulses}"

    def test_no_nans(self):
        _, x = self._simulate(log_rt=True)
        assert not torch.any(torch.isnan(x)), "NaN in training data"

    def test_no_infs(self):
        _, x = self._simulate(log_rt=True)
        assert not torch.any(torch.isinf(x)), "Inf in training data"


# Timeout handling logic tests
class TestTimeoutHandling:

    def test_forced_trials_have_tmax_rt(self, device):
        """
        With a very large boundary and small drift, most trials time out.
        After forcing, their raw RT should be T_MAX.
        """
        prior = build_prior_theta()
        theta_hard = torch.tensor([0.5, 0.1, 0.01, 100.0, 0.0])

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, x = simulate_training_sessions(
                prior,
                num_sessions=1,
                num_trials=16,
                simulate_batch_fn=simulate_rt_choice_batch,
                device=device,
                mu_sensory=1.0,
                p_success=0.7,
                P=P_MAX,
                log_rt=False,  # raw RT so can check T_MAX directly
                seed=0,
                theta=theta_hard,
            )
        x_3d = x.view(1, 16, TRIAL_DIM)[0]
        rt = x_3d[:, 0]
        # At least some should be exactly T_MAX (forced)
        n_forced = (rt >= T_MAX - 0.01).sum().item()
        assert n_forced > 0, "Expected some forced T_MAX trials with extreme theta"
