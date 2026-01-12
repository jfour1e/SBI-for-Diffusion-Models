import os
import numpy as np
import torch
from torch.distributions import Distribution

from sbi_for_diffusion_models.models.rt_choice_model import generate_pulse_matrix_numpy

# TRAINING PROPOSAL
class PulseSequenceProposal(Distribution):
    """
    Proposal distribution over pulse sequences of length P.

    Only sampling is needed for MNLE training.
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(self, P: int, p_success: float, seed: int = 0, device=None):
        super().__init__(validate_args=False)
        self.P = int(P)
        self.p_success = float(p_success)
        self.rng = np.random.default_rng(seed)
        self._device = device

    @property
    def event_shape(self):
        return torch.Size([self.P])

    def sample(self, sample_shape=torch.Size()):
        n = int(np.prod(sample_shape)) if len(sample_shape) > 0 else 1
        s_np = generate_pulse_matrix_numpy(
            self.rng, n_trials=n, n_pulses=self.P, p_success=self.p_success
        )
        s = torch.from_numpy(s_np).to(dtype=torch.float32)
        if len(sample_shape) > 0:
            s = s.view(*sample_shape, self.P)
        if self._device is not None:
            s = s.to(self._device)
        return s

    def log_prob(self, value):
        # Not needed for training, but keep defined for completeness.
        return torch.zeros(value.shape[:-1], device=value.device, dtype=torch.float32)


class ExtendedProposal(Distribution):
    """Proposal over concatenated z=[theta(5), pulse_sides(P)] used to train MNLE."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, theta_prior: Distribution, pulse_proposal: PulseSequenceProposal, device=None):
        super().__init__(validate_args=False)
        self.theta_prior = theta_prior
        self.pulse_proposal = pulse_proposal
        self._device = device

    @property
    def event_shape(self):
        return torch.Size([5 + self.pulse_proposal.P])

    def sample(self, sample_shape=torch.Size()):
        theta = self.theta_prior.sample(sample_shape)
        pulses = self.pulse_proposal.sample(sample_shape)
        z = torch.cat([theta.to(torch.float32), pulses.to(torch.float32)], dim=-1)
        if self._device is not None:
            z = z.to(self._device)
        return z

    def log_prob(self, z):
        theta = z[..., :5]
        pulses = z[..., 5:]
        return self.theta_prior.log_prob(theta) + self.pulse_proposal.log_prob(pulses)

