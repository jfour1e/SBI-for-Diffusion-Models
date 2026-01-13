import os
import numpy as np
import torch
from torch.distributions import Distribution

# Posterior potential over theta only (adds log prior; conditioned loglike is likelihood only).
class ThetaOnlyPosteriorPotential:
    def __init__(
        self,
        *,
        conditioned_loglike,
        prior_theta: Distribution,
        x_o: torch.Tensor,
        device: str = "cpu",
        temperature: float = 1.0,
    ):
        self.conditioned_loglike = conditioned_loglike
        self.prior_theta = prior_theta
        self._x_o = x_o.to(device=device, dtype=torch.float32)
        self.device = device
        self.temperature = float(temperature)

    def return_x_o(self):
        return self._x_o

    def set_x_o(self, x_o: torch.Tensor):
        self._x_o = x_o.to(self.device, dtype=torch.float32)
        return self

    def set_x(self, x: torch.Tensor):
        return self.set_x_o(x)

    def __call__(self, theta: torch.Tensor, x_o: torch.Tensor = None, track_gradients: bool = True) -> torch.Tensor:
        # IMPORTANT: sbi may call potential(theta, x_o). If provided, update internal x.
        if x_o is not None:
            self.set_x_o(x_o)

        if theta.ndim == 1:
            theta = theta.view(1, -1)
        theta = theta.to(self.device, dtype=torch.float32)

        # Prior term
        lp = self.prior_theta.log_prob(theta)  # (N,)
        valid = torch.isfinite(lp)
        if not torch.any(valid):
            return lp

        # Likelihood term (conditioned on pulses via condition_on_theta)
        with torch.set_grad_enabled(bool(track_gradients)):
            ll = self.conditioned_loglike(
                theta[valid], 
                self._x_o, 
                track_gradients=bool(track_gradients)).reshape(-1)

        out = lp.clone()
        out[valid] = out[valid] + ll / self.temperature
        return out

class ConditionedMNLELogLikelihood(torch.nn.Module):
    """
    Pickleable replacement for LikelihoodBasedPotential.condition_on_theta(...).

    Computes sum_i log p(x_i | global_theta, local_theta_i) efficiently
    by moving iid trials onto the batch dimension of theta, following
    sbi's _log_likelihood_over_iid_trials_and_local_theta implementation.
    """

    def __init__(self, estimator, local_theta: torch.Tensor, device: str = "cpu"):
        super().__init__()
        self.estimator = estimator
        self.device = device
        # store as buffer so it moves with .to(...) and is pickleable
        self.register_buffer("local_theta", local_theta.to(device=device, dtype=torch.float32))

    def forward(
        self,
        global_theta: torch.Tensor,  # (N, 5)
        x_o: torch.Tensor,           # (num_trials, 2) or (num_trials, 1, 2)
        track_gradients: bool = True,
    ) -> torch.Tensor:
        global_theta = global_theta.to(self.device, dtype=torch.float32)
        x_o = x_o.to(self.device, dtype=torch.float32)

        # Ensure x has shape (num_trials, num_xs=1, event_dim=2)
        if x_o.dim() == 2:
            x = x_o.unsqueeze(1)  # (T,1,2)
        else:
            x = x_o

        num_trials, num_xs = x.shape[:2]
        assert num_xs == 1, "This implementation supports a single observed x batch (num_xs=1)."
        assert self.local_theta.shape[0] == num_trials, (
            f"local_theta must have shape (num_trials, P). Got {tuple(self.local_theta.shape)}"
        )

        num_thetas = global_theta.shape[0]

        # Following sbi: move iid trials onto batch dim of theta and repeat there
        # x_repeated shape: (1, num_trials*num_thetas, 2)
        x_repeated = torch.transpose(x, 0, 1).repeat_interleave(num_thetas, dim=1)

        # Build condition tensor [global_theta, local_theta_i] for each trial-theta pair
        # theta_with_condition shape: (num_trials*num_thetas, 5+P)
        theta_with_condition = torch.cat(
            [
                global_theta.repeat(num_trials, 1),                      # ABAB...
                self.local_theta.repeat_interleave(num_thetas, dim=0),   # AABB...
            ],
            dim=-1,
        )

        with torch.set_grad_enabled(bool(track_gradients)):
            ll_batch = self.estimator.log_prob(x_repeated, condition=theta_with_condition)
            # reshape to (num_xs=1, num_trials, num_thetas) and sum over trials
            ll_sum = ll_batch.reshape(num_xs, num_trials, num_thetas).sum(1).squeeze(0)

        return ll_sum  # (num_thetas,)

