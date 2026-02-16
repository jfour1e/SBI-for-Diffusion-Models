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
            ExpNormal(torch.tensor([-1.0]), torch.tensor([0.5])),       # lam ∈ (0,∞)
            ExpNormal(torch.tensor([0.0]), torch.tensor([0.5])),        # v   ∈ (0,∞)
            ExpNormal(torch.tensor([2.00]), torch.tensor([0.5])),       # B   ∈ (0,∞)
            LogisticNormal(torch.tensor([0.0]), torch.tensor([1.5])),   # tau ∈ (0,1)
        ]
    )
