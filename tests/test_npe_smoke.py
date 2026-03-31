"""
NPE training smoke tests.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import pytest
import torch

from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    max_num_pulses,
)
from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
)
from sbi_for_diffusion_models.priors import build_prior_theta, build_prior_theta_lapse
from sbi_for_diffusion_models.mnpe import (
    train_npe_session,
    _build_npe_embedding_net,
    _build_npe_estimator_builder,
    _simulate_dummy_batch,
)
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions

P_MAX = max_num_pulses()
TRIAL_DIM = 2 + P_MAX


# Small RunConfig that keeps the smoke test fast
@dataclass(frozen=True)
class _TinyConfig:
    MU_SENSORY: float = 1.0
    P_SUCCESS: float = 0.7
    NUM_TRIALS_OBS: int = 16        
    LOG_RT_MANUALLY: bool = True

    NPE_NUM_SESSIONS: int = 4
    NPE_TRAIN_BATCH_SIZE: int = 4
    NPE_HIDDEN_FEATURES: int = 32
    NPE_NUM_TRANSFORMS: int = 2
    NPE_NUM_BINS: int = 4
    NPE_SESSIONS_PER_STEP: int = 4    
    NPE_NUM_STEPS: int = 5            # 
    NPE_LR: float = 1e-3

    NPE_TRIAL_NET_HIDDEN: int = 32
    NPE_TRIAL_NET_LAYERS: int = 2
    NPE_TRIAL_NET_OUTPUT_DIM: int = 16
    NPE_AGG_FN: str = "mean"
    NPE_POST_AGG_HIDDEN: int = 32
    NPE_POST_AGG_LAYERS: int = 1
    NPE_EMBEDDING_OUTPUT_DIM: int = 16

    # Inference
    NPE_POSTERIOR_SAMPLES: int = 50

    NPE_PATIENCE: int = 9999
    NPE_MIN_DELTA: float = 0.0
    NPE_EMA_BETA: float = 0.98

TINY_CFG = _TinyConfig()

# Base model (5 params)
class TestNPESmokeBase:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cfg = TINY_CFG
        self.prior = build_prior_theta()
        self.sim_fn = simulate_rt_choice_batch
        self.device = "cpu"
        self.theta_dim = 5

    def test_training_loop_runs(self):
        """train_npe_session should complete without error."""
        density_est, posterior = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        assert density_est is not None
        assert posterior is not None

    def test_loss_is_finite(self):
        """Every loss value during training should be finite."""
        density_est, _ = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        # Run one more forward pass to spot-check
        T = int(self.cfg.NUM_TRIALS_OBS)
        theta_b, x_b = simulate_training_sessions(
            self.prior,
            num_sessions=2,
            num_trials=T,
            simulate_batch_fn=self.sim_fn,
            device=torch.device(self.device),
            mu_sensory=float(self.cfg.MU_SENSORY),
            p_success=float(self.cfg.P_SUCCESS),
            P=P_MAX,
            log_rt=bool(self.cfg.LOG_RT_MANUALLY),
            seed=999,
        )
        density_est.eval()
        with torch.no_grad():
            losses = density_est.loss(theta_b, condition=x_b)
        assert torch.all(torch.isfinite(losses)), f"Non-finite loss: {losses}"

    def test_posterior_sample_shape(self):
        _, posterior = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        T = int(self.cfg.NUM_TRIALS_OBS)
        _, x_o = simulate_training_sessions(
            self.prior,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=self.sim_fn,
            device=torch.device(self.device),
            mu_sensory=float(self.cfg.MU_SENSORY),
            p_success=float(self.cfg.P_SUCCESS),
            P=P_MAX,
            log_rt=bool(self.cfg.LOG_RT_MANUALLY),
            seed=123,
        )
        n_samples = 20
        samples = posterior.sample(
            (n_samples,), x=x_o, show_progress_bars=False,
        ).detach().cpu()
        assert samples.shape == (n_samples, self.theta_dim)

    def test_posterior_samples_finite(self):
        _, posterior = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        T = int(self.cfg.NUM_TRIALS_OBS)
        _, x_o = simulate_training_sessions(
            self.prior,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=self.sim_fn,
            device=torch.device(self.device),
            mu_sensory=float(self.cfg.MU_SENSORY),
            p_success=float(self.cfg.P_SUCCESS),
            P=P_MAX,
            log_rt=bool(self.cfg.LOG_RT_MANUALLY),
            seed=123,
        )
        samples = posterior.sample(
            (20,), x=x_o, show_progress_bars=False,
        ).detach().cpu()
        assert torch.all(torch.isfinite(samples)), "Non-finite posterior samples"

    def test_checkpoint_roundtrip(self):
        """Save and reload weights — the density estimator should produce identical loss."""
        density_est, _ = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "npe_test.pt")
            torch.save(
                {"state_dict": density_est.state_dict(), "config": self.cfg},
                path,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Rebuild a fresh estimator with the same architecture
        T = int(self.cfg.NUM_TRIALS_OBS)
        embedding = _build_npe_embedding_net(self.cfg, T=T, P=P_MAX)
        builder = _build_npe_estimator_builder(self.cfg, embedding)

        theta_d, x_d = _simulate_dummy_batch(
            self.cfg, self.prior,
            simulate_batch_fn=self.sim_fn,
            dev=torch.device(self.device), seed=0, T=T, P=P_MAX,
        )
        est_reloaded = builder(theta_d, x_d)
        est_reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
        est_reloaded.eval()
        density_est.eval()

        # Both should produce the same loss on identical data
        with torch.no_grad():
            loss_orig = density_est.loss(theta_d, condition=x_d).mean()
            loss_reload = est_reloaded.loss(theta_d, condition=x_d).mean()
        assert torch.allclose(loss_orig, loss_reload, atol=1e-5), (
            f"Checkpoint mismatch: {loss_orig:.6f} vs {loss_reload:.6f}"
        )


# Lapse model (6 params) smoke test
class TestNPESmokeLapse:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cfg = TINY_CFG
        self.prior = build_prior_theta_lapse()
        self.sim_fn = simulate_rt_choice_batch_lapse
        self.device = "cpu"
        self.theta_dim = 6

    def test_training_loop_runs(self):
        density_est, posterior = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        assert density_est is not None
        assert posterior is not None

    def test_posterior_sample_shape(self):
        _, posterior = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        T = int(self.cfg.NUM_TRIALS_OBS)
        _, x_o = simulate_training_sessions(
            self.prior,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=self.sim_fn,
            device=torch.device(self.device),
            mu_sensory=float(self.cfg.MU_SENSORY),
            p_success=float(self.cfg.P_SUCCESS),
            P=P_MAX,
            log_rt=bool(self.cfg.LOG_RT_MANUALLY),
            seed=42,
        )
        samples = posterior.sample(
            (20,), x=x_o, show_progress_bars=False,
        ).detach().cpu()
        assert samples.shape == (20, self.theta_dim)
        assert torch.all(torch.isfinite(samples))

    def test_loss_is_finite(self):
        density_est, _ = train_npe_session(
            self.cfg,
            self.prior,
            simulate_batch_fn=self.sim_fn,
            device=self.device,
            seed=0,
        )
        T = int(self.cfg.NUM_TRIALS_OBS)
        theta_b, x_b = simulate_training_sessions(
            self.prior,
            num_sessions=2,
            num_trials=T,
            simulate_batch_fn=self.sim_fn,
            device=torch.device(self.device),
            mu_sensory=float(self.cfg.MU_SENSORY),
            p_success=float(self.cfg.P_SUCCESS),
            P=P_MAX,
            log_rt=bool(self.cfg.LOG_RT_MANUALLY),
            seed=999,
        )
        density_est.eval()
        with torch.no_grad():
            losses = density_est.loss(theta_b, condition=x_b)
        assert torch.all(torch.isfinite(losses))


#  Embedding - density estimator compatibility
class TestEmbeddingEstimatorCompat:
    """
    Verify the embedding net and the NSF density estimator agree on
    dimensions so that no shape mismatch occurs at the junction.
    """

    def test_dummy_forward_pass(self):
        cfg = TINY_CFG
        prior = build_prior_theta()
        T = int(cfg.NUM_TRIALS_OBS)

        embedding = _build_npe_embedding_net(cfg, T=T, P=P_MAX)
        builder = _build_npe_estimator_builder(cfg, embedding)

        theta_d, x_d = _simulate_dummy_batch(
            cfg, prior,
            simulate_batch_fn=simulate_rt_choice_batch,
            dev=torch.device("cpu"), seed=0, T=T, P=P_MAX,
        )
        estimator = builder(theta_d, x_d)
        estimator.eval()

        with torch.no_grad():
            losses = estimator.loss(theta_d, condition=x_d)
        assert losses.shape == (theta_d.shape[0],)
        assert torch.all(torch.isfinite(losses))

    def test_lapse_dummy_forward_pass(self):
        cfg = TINY_CFG
        prior = build_prior_theta_lapse()
        T = int(cfg.NUM_TRIALS_OBS)

        embedding = _build_npe_embedding_net(cfg, T=T, P=P_MAX)
        builder = _build_npe_estimator_builder(cfg, embedding)

        theta_d, x_d = _simulate_dummy_batch(
            cfg, prior,
            simulate_batch_fn=simulate_rt_choice_batch_lapse,
            dev=torch.device("cpu"), seed=0, T=T, P=P_MAX,
        )
        estimator = builder(theta_d, x_d)
        estimator.eval()

        with torch.no_grad():
            losses = estimator.loss(theta_d, condition=x_d)
        assert losses.shape == (theta_d.shape[0],)
        assert torch.all(torch.isfinite(losses))
