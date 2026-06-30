"""
Prior distributions for the RT-choice model.

Custom wrappers (LogisticNormal, ExpNormal) around TransformedDistribution
that are compatible with sbi's MultipleIndependent.to(device).
"""
from __future__ import annotations

import torch
from torch.distributions import (
    Distribution,
    Normal,
    TransformedDistribution,
    constraints,
)


def _ar_marginal() -> Normal:
    """Normal(0, 0.25) on w_corr / w_err: zero-centered (no AR effect),
    tail mostly within ±0.5; simulator clamps a0_eff to [0,1] anyway."""
    return Normal(torch.tensor([0.0]), torch.tensor([0.25]))
from torch.distributions.transforms import ExpTransform, SigmoidTransform
from sbi.utils import MultipleIndependent


class _SbiCompatibleTransformed(Distribution):
    """
    Base for TransformedDistribution wrappers that survive
    ``MultipleIndependent.to(device)``.

    sbi reconstructs distributions via
    ``type(dist)(**{k: v.to(dev) for k,v in dist.__dict__ if tensor})``.
    Raw TransformedDistribution breaks because its ctor needs positional args.
    These subclasses store ``loc`` and ``scale`` as direct tensor attrs so the
    reconstruction round-trips correctly.
    """

    has_rsample = True

    # Subclasses set _transform
    _transform: torch.distributions.Transform

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self._inner = TransformedDistribution(
            Normal(loc, scale), self._transform
        )
        super().__init__(
            self._inner.batch_shape, self._inner.event_shape, validate_args
        )

    def rsample(self, sample_shape=torch.Size()):
        return self._inner.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self._inner.sample(sample_shape)

    def log_prob(self, value):
        return self._inner.log_prob(value)


class LogisticNormal(_SbiCompatibleTransformed):
    """Normal in unconstrained space → Sigmoid → (0, 1)."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.unit_interval
    _transform = SigmoidTransform()


class ExpNormal(_SbiCompatibleTransformed):
    """Normal in unconstrained space → Exp → (0, ∞)."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    _transform = ExpTransform()


def build_prior_theta() -> Distribution:
    """
    Prior over theta = [a0, lam, v, B, tau].

    All priors are Normal in unconstrained space, pushed through Sigmoid (for
    [0,1]-bounded params) or Exp (for positive params).  This gives the MCMC
    sampler Gaussian geometry in the unconstrained space.
    """
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),   # a0  ∈ (0,1)
            ExpNormal(torch.tensor([-1.5]), torch.tensor([0.35])),      # lam ∈ (0,∞)  loc -1.0→-1.5, scale 0.5→0.35: mean ~0.24, avoids high-lam timeouts with OU→B/2
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),        # v   ∈ (0,∞)
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),       # B   ∈ (0,∞)  scale 0.5→0.35: tighter tail, mean ~2.9, avoids high-B timeouts with OU→B/2
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),   # tau ∈ (0,1)
        ]
    )

# for new lapse model
def build_prior_theta_lapse() -> Distribution:
    """
    Prior over theta = [a0, lam, v, B, tau, p_lapse].

    All priors are Normal in unconstrained space, pushed through Sigmoid (for
    [0,1]-bounded params) or Exp (for positive params).  This gives the MCMC
    sampler Gaussian geometry in the unconstrained space.
    """
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),   # a0  ∈ (0,1)
            ExpNormal(torch.tensor([-1.5]), torch.tensor([0.35])),      # lam ∈ (0,∞)  loc -1.0→-1.5, scale 0.5→0.35: mean ~0.24, avoids high-lam timeouts with OU→B/2
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),        # v   ∈ (0,∞)
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),       # B   ∈ (0,∞)  scale 0.5→0.35: tighter tail, mean ~2.9, avoids high-B timeouts with OU→B/2
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),   # tau ∈ (0,1)
            LogisticNormal(torch.tensor([-3.0]), torch.tensor([0.7])), # lapse ∈ (0,1)
        ]
    )


def build_prior_theta_noleak() -> Distribution:
    """
    Prior over theta = [a0, v, B, tau] (no leak; pure Brownian accumulator).

    Drops `lam` from build_prior_theta; remaining marginals are identical.
    """
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),   # a0  ∈ (0,1)
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),        # v   ∈ (0,∞)
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),       # B   ∈ (0,∞)
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),   # tau ∈ (0,1)
        ]
    )


def build_prior_theta_lapse_noleak() -> Distribution:
    """
    Prior over theta = [a0, v, B, tau, p_lapse] (no leak; pure Brownian + lapse).

    Drops `lam` from build_prior_theta_lapse; remaining marginals are identical.
    """
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),   # a0  ∈ (0,1)
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),        # v   ∈ (0,∞)
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),       # B   ∈ (0,∞)
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),   # tau ∈ (0,1)
            LogisticNormal(torch.tensor([-3.0]), torch.tensor([0.7])),  # lapse ∈ (0,1)
        ]
    )


def build_prior_theta_ar() -> Distribution:
    """Prior over theta = [a0, lam, v, B, tau, w_corr, w_err]."""
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([-1.5]), torch.tensor([0.35])),
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),
            _ar_marginal(),  # w_corr
            _ar_marginal(),  # w_err
        ]
    )


def build_prior_theta_noleak_ar() -> Distribution:
    """Prior over theta = [a0, v, B, tau, w_corr, w_err]."""
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),
            _ar_marginal(),
            _ar_marginal(),
        ]
    )


def build_prior_theta_lapse_ar() -> Distribution:
    """Prior over theta = [a0, lam, v, B, tau, p_lapse, w_corr, w_err]."""
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([-1.5]), torch.tensor([0.35])),
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),
            LogisticNormal(torch.tensor([-3.0]), torch.tensor([0.7])),
            _ar_marginal(),
            _ar_marginal(),
        ]
    )


def build_prior_theta_lapse_noleak_ar() -> Distribution:
    """Prior over theta = [a0, v, B, tau, p_lapse, w_corr, w_err]."""
    return MultipleIndependent(
        [
            LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),
            ExpNormal(torch.tensor([1.0]), torch.tensor([0.35])),
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),
            LogisticNormal(torch.tensor([-3.0]), torch.tensor([0.7])),
            _ar_marginal(),
            _ar_marginal(),
        ]
    )


# ---------------------------------------------------------------------------
# Mouse-retuned priors
#
# The mouse task runs on a faster timescale than the marmoset/human task
# (100 ms pulses, T_MAX = 5 s) and the animals respond quickly (median RT
# ~0.4 s ≈ 4 pulses). The marmoset priors place too much mass on a large
# non-decision time `tau` (median 0.5 s) and a large boundary `B` (median
# ~2.7), which over-predicts slow RTs. The mouse marginals below are shifted:
#   tau : LogisticNormal(-1.95, 0.5)  -> median ~0.13 s, 95th ~0.24 s (< typical RT)
#   B   : ExpNormal(0.4, 0.35)        -> median ~1.5 (fewer pulses to a decision)
#   v   : ExpNormal(-1.2, 0.5)        -> median ~0.30 per-pulse kick
#   a0  : LogisticNormal(0, 0.5)      -> unbiased start (unchanged)
#   p_lapse : LogisticNormal(-3, 0.7) -> median ~0.05 (unchanged)
#   w_* : Normal(0, 0.25)             -> zero-centered AR bias (unchanged)
# These are starting values, validated against the real mouse RT/accuracy
# distributions by scripts/prior_predictive_check.py (SPECIES=mouse).
# ---------------------------------------------------------------------------

def _mouse_a0() -> Distribution:
    return LogisticNormal(torch.tensor([0.0]), torch.tensor([0.5]))


def _mouse_v() -> Distribution:
    return ExpNormal(torch.tensor([-1.2]), torch.tensor([0.5]))


def _mouse_B() -> Distribution:
    return ExpNormal(torch.tensor([0.4]), torch.tensor([0.35]))


def _mouse_tau() -> Distribution:
    return LogisticNormal(torch.tensor([-1.95]), torch.tensor([0.5]))


def _mouse_p_lapse() -> Distribution:
    return LogisticNormal(torch.tensor([-3.0]), torch.tensor([0.7]))


def build_prior_theta_lapse_noleak_mouse() -> Distribution:
    """Mouse prior over theta = [a0, v, B, tau, p_lapse] (no leak + lapse)."""
    return MultipleIndependent(
        [_mouse_a0(), _mouse_v(), _mouse_B(), _mouse_tau(), _mouse_p_lapse()]
    )


def build_prior_theta_lapse_noleak_ar_mouse() -> Distribution:
    """Mouse prior over theta = [a0, v, B, tau, p_lapse, w_corr, w_err]."""
    return MultipleIndependent(
        [
            _mouse_a0(), _mouse_v(), _mouse_B(), _mouse_tau(), _mouse_p_lapse(),
            _ar_marginal(),  # w_corr
            _ar_marginal(),  # w_err
        ]
    )

