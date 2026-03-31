"""
Tests for prior distributions and the permutation-invariant embedding network.

Covers:
  - Prior sampling shapes and support constraints
  - log_prob is finite for in-support samples
  - LogisticNormal ∈ (0,1), ExpNormal ∈ (0,∞)
  - Embedding network input/output shapes
  - Permutation invariance of the embedding
  - Embedding handles edge-case inputs (zeros, large values)
"""
from __future__ import annotations

import pytest
import torch

from sbi_for_diffusion_models.priors import (
    build_prior_theta,
    build_prior_theta_lapse,
    LogisticNormal,
    ExpNormal,
)
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses

P_MAX = max_num_pulses()
TRIAL_DIM = 2 + P_MAX

# Prior test
class TestBuildPriorTheta:

    def test_sample_shape(self):
        prior = build_prior_theta()
        samples = prior.sample((100,))
        assert samples.shape == (100, 5)

    def test_sample_support(self):
        prior = build_prior_theta()
        samples = prior.sample((500,))
        a0, lam, v, B, tau = [samples[:, i] for i in range(5)]
        assert torch.all(a0 > 0) and torch.all(a0 < 1), "a0 must be in (0,1)"
        assert torch.all(lam > 0), "lam must be positive"
        assert torch.all(v > 0), "v must be positive"
        assert torch.all(B > 0), "B must be positive"
        assert torch.all(tau > 0) and torch.all(tau < 1), "tau must be in (0,1)"

    def test_log_prob_finite(self):
        prior = build_prior_theta()
        samples = prior.sample((50,))
        lp = prior.log_prob(samples)
        assert torch.all(torch.isfinite(lp)), "Non-finite log_prob for valid samples"

    def test_single_sample(self):
        prior = build_prior_theta()
        s = prior.sample((1,))
        assert s.shape == (1, 5)

class TestBuildPriorThetaLapse:

    def test_sample_shape(self):
        prior = build_prior_theta_lapse()
        samples = prior.sample((100,))
        assert samples.shape == (100, 6), "Lapse prior should have 6 params"

    def test_lapse_in_unit_interval(self):
        prior = build_prior_theta_lapse()
        samples = prior.sample((500,))
        p_lapse = samples[:, 5]
        assert torch.all(p_lapse > 0) and torch.all(p_lapse < 1)


class TestCustomDistributions:

    def test_logistic_normal_in_01(self):
        d = LogisticNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        samples = d.sample((1000,))
        assert torch.all(samples > 0) and torch.all(samples < 1)

    def test_exp_normal_positive(self):
        d = ExpNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        samples = d.sample((1000,))
        assert torch.all(samples > 0)

    def test_logistic_normal_log_prob(self):
        d = LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5]))
        s = d.sample((10,))
        lp = d.log_prob(s)
        assert torch.all(torch.isfinite(lp))

    def test_exp_normal_log_prob(self):
        d = ExpNormal(torch.tensor([0.0]), torch.tensor([0.5]))
        s = d.sample((10,))
        lp = d.log_prob(s)
        assert torch.all(torch.isfinite(lp))


# Test embedding network 
class TestPermutationInvariantEmbedding:

    @pytest.fixture
    def embedding(self):
        return PermutationInvariantEmbedding(
            num_trials=32,
            trial_dim=TRIAL_DIM,
            trial_net_hidden=32,
            trial_net_layers=2,
            trial_net_output_dim=16,
            post_agg_hidden=32,
            post_agg_layers=1,
            output_dim=16,
            aggregation="mean",
        )

    def test_output_shape(self, embedding):
        B, T = 4, 32
        x = torch.randn(B, T * TRIAL_DIM)
        out = embedding(x)
        assert out.shape == (B, 16)

    def test_single_batch(self, embedding):
        x = torch.randn(1, 32 * TRIAL_DIM)
        out = embedding(x)
        assert out.shape == (1, 16)

    def test_permutation_invariance(self, embedding):
        """Shuffling trial order should give the same output (with mean aggregation)."""
        T = 32
        torch.manual_seed(123)
        x_3d = torch.randn(1, T, TRIAL_DIM)
        x_flat = x_3d.view(1, -1)

        # Permute trials
        perm = torch.randperm(T)
        x_perm_3d = x_3d[:, perm, :]
        x_perm_flat = x_perm_3d.view(1, -1)

        embedding.eval()
        with torch.no_grad():
            out_orig = embedding(x_flat)
            out_perm = embedding(x_perm_flat)

        assert torch.allclose(out_orig, out_perm, atol=1e-5), (
            "Embedding is not permutation invariant"
        )

    def test_sum_aggregation(self):
        emb = PermutationInvariantEmbedding(
            num_trials=16,
            trial_dim=TRIAL_DIM,
            trial_net_hidden=16,
            trial_net_layers=1,
            trial_net_output_dim=8,
            post_agg_hidden=16,
            post_agg_layers=1,
            output_dim=8,
            aggregation="sum",
        )
        x = torch.randn(2, 16 * TRIAL_DIM)
        out = emb(x)
        assert out.shape == (2, 8)

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            PermutationInvariantEmbedding(
                num_trials=16,
                trial_dim=TRIAL_DIM,
                aggregation="max",
            )

    def test_zero_input(self, embedding):
        x = torch.zeros(1, 32 * TRIAL_DIM)
        out = embedding(x)
        assert torch.all(torch.isfinite(out))

    def test_gradients_flow(self, embedding):
        x = torch.randn(2, 32 * TRIAL_DIM, requires_grad=True)
        out = embedding(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)
