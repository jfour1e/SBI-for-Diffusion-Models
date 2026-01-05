import torch
torch.distributions.Distribution.set_default_validate_args(True)
import matplotlib.pyplot as plt
import numpy as np
import math

from torch.distributions import Normal, Beta, LogNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch.distributions import Distribution

from sbi.analysis import pairplot
from sbi.inference import MNLE, MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.utils.get_nn_models import likelihood_nn
from sbi.utils import mcmc_transform

# IMPORTANT: you need the refactored simulator that accepts pulse_sides
from sbi_for_diffusion_models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    pulse_schedule,
    n_pulses_max_from_schedule,
    generate_pulse_matrix_numpy,
)

PULSE_INTERVAL = 0.1  # seconds


# ----------------------------
# Prior over global parameters theta = [a0, lam, v, B, tnd]
# ----------------------------
class JointPrior(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, a0_prior, lam_prior, v_prior, B_prior, tnd_prior, device=None):
        super().__init__(validate_args=False)
        self.a0_prior = a0_prior
        self.lam_prior = lam_prior
        self.v_prior = v_prior
        self.B_prior = B_prior
        self.tnd_prior = tnd_prior
        self._device = device

    @property
    def event_shape(self):
        return torch.Size([5])

    def sample(self, sample_shape=torch.Size()):
        a0  = self.a0_prior.sample(sample_shape)
        lam = self.lam_prior.sample(sample_shape)
        v   = self.v_prior.sample(sample_shape)
        B   = self.B_prior.sample(sample_shape)
        tnd = self.tnd_prior.sample(sample_shape)
        theta = torch.stack([a0, lam, v, B, tnd], dim=-1)
        if self._device is not None:
            theta = theta.to(self._device)
        return theta

    def log_prob(self, theta):
        if theta.shape[-1] != 5:
            raise ValueError(f"Expected theta[...,5], got {theta.shape}")

        if self._device is not None and theta.device != self._device:
            theta = theta.to(self._device)

        a0, lam, v, B, tnd = theta.unbind(-1)

        ok = (
            (a0 >= 0.0) & (a0 <= 1.0) &
            (lam > 0.0) &
            (v > 0.0) &
            (B > 0.0) &
            (tnd >= 0.0) & (tnd <= 0.9)
        )

        lp = torch.full(a0.shape, -torch.inf, device=theta.device)

        if ok.any():
            lp_ok = (
                self.a0_prior.log_prob(a0[ok])
                + self.lam_prior.log_prob(lam[ok])
                + self.v_prior.log_prob(v[ok])
                + self.B_prior.log_prob(B[ok])
                + self.tnd_prior.log_prob(tnd[ok])
            )
            lp[ok] = lp_ok

        return lp


# ----------------------------
# Pulse-condition "prior" (proposal) for training
# ----------------------------
class PulseSidePrior(Distribution):
    arg_constraints = {}
    has_rsample = False

    def __init__(self, P: int, p_success: float, seed: int = 0, device=None):
        super().__init__(validate_args=False)
        self.P = int(P)
        self.p = float(p_success)
        self.rng = np.random.default_rng(seed)
        self._device = device

        # precompute logs
        eps = 1e-12
        p = min(max(self.p, eps), 1.0 - eps)
        self._logp = float(math.log(p))
        self._log1mp = float(math.log(1.0 - p))
        self._loghalf = float(math.log(0.5))

    @property
    def event_shape(self):
        return torch.Size([self.P])

    def sample(self, sample_shape=torch.Size()):
        n = int(np.prod(sample_shape)) if len(sample_shape) > 0 else 1
        s_np = generate_pulse_matrix_numpy(self.rng, n_trials=n, n_pulses=self.P, p_success=self.p)
        s = torch.from_numpy(s_np).to(dtype=torch.float32)
        if len(sample_shape) > 0:
            s = s.view(*sample_shape, self.P)
        if self._device is not None:
            s = s.to(self._device)
        return s

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # value shape (..., P), entries expected in {-1, +1}
        if value.shape[-1] != self.P:
            raise ValueError(f"Expected value[..., {self.P}], got {tuple(value.shape)}")

        v = value
        if self._device is not None and v.device != self._device:
            v = v.to(self._device)
        v = v.to(torch.float32)

        # Count +1's per sequence (k)
        k = (v > 0).sum(dim=-1).to(torch.float32)   # shape (...)

        P = float(self.P)
        # log term if correct_side = +1: +1 with prob p, -1 with prob 1-p
        log_a = self._loghalf + k * self._logp + (P - k) * self._log1mp
        # log term if correct_side = -1: +1 with prob 1-p, -1 with prob p
        log_b = self._loghalf + k * self._log1mp + (P - k) * self._logp

        return torch.logsumexp(torch.stack([log_a, log_b], dim=0), dim=0)


class ExtendedProposal(Distribution):
    """
    Proposal over concatenated [theta(5), pulse_sides(P)] used to train MNLE.
    """
    arg_constraints = {}
    has_rsample = False

    def __init__(self, theta_prior: JointPrior, pulse_proposal: PulseSidePrior, device=None):
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

# ----------------------------
# Pack x for MNLE: x = [rt, choice], with choice LAST, changed to let SBI do the log transform 
# ----------------------------
def pack_x_rt_choice(rt_choice: torch.Tensor) -> torch.Tensor:
    rt = rt_choice[:, 0:1].to(torch.float32)             # raw rt
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)


# ----------------------------
# Simulator wrapper: (theta_and_pulses) -> x = [rt, choice]
# ----------------------------
@torch.no_grad()
def sim_wrapper(theta_and_pulses: torch.Tensor, *, mu_sensory: float, p_success: float, P: int):
    """
    theta_and_pulses: (N, 5+P) where first 5 dims are global theta, remaining P are pulse_sides
    returns x: (N,2) = [rt, choice]
    """
    theta = theta_and_pulses[:, :5]
    pulse_sides = theta_and_pulses[:, 5:5 + P]

    rt_choice = rt_choice_model_simulator_torch(
        theta,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,  # not used if pulse_sides provided; safe
    )
    return pack_x_rt_choice(rt_choice)


# ----------------------------
# Training set generation: (theta+pulses) -> x
# ----------------------------
@torch.no_grad()
def simulate_training_set_with_conditions(
    proposal: ExtendedProposal,
    num_simulations: int,
    batch_size: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
):
    zs = []
    xs = []

    for start in range(0, num_simulations, batch_size):
        curr_batches = min(batch_size, num_simulations - start)

        z = proposal.sample((curr_batches,)).to(device=device, dtype=torch.float32)
        x = sim_wrapper(z, mu_sensory=mu_sensory, p_success=p_success, P=P)

        zs.append(z.detach().cpu())
        xs.append(x.detach().cpu())

    z_cpu = torch.cat(zs, dim=0).to(torch.float32)
    x_all = torch.cat(xs, dim=0).to(torch.float32)

    assert torch.isfinite(z_cpu).all()
    assert torch.isfinite(x_all).all()
    assert torch.all((x_all[:, -1] == 0) | (x_all[:, -1] == 1) | (x_all[:, -1] == 2))

    print("Training x shape:", tuple(x_all.shape), " (should be (N,2) = [rt, choice])")
    print("Training z shape:", tuple(z_cpu.shape), " (should be (N, 5+P) = [theta, pulses])")
    print("Unique outcomes in training (choice):", x_all[:, -1].unique().tolist())
    return z_cpu, x_all


# ----------------------------
# Observed data generation (for demo)
# ----------------------------
@torch.no_grad()
def simulate_observed_session(
    theta_true: torch.Tensor,
    num_trials: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    s_np = generate_pulse_matrix_numpy(rng, n_trials=num_trials, n_pulses=P, p_success=p_success)
    pulse_sides = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

    theta_rep = theta_true.view(1, 5).repeat(num_trials, 1)
    rt_choice = rt_choice_model_simulator_torch(
        theta_rep,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,
    )
    x_o = pack_x_rt_choice(rt_choice)

    print("Observed x shape:", tuple(x_o.shape), " (N,2)")
    print("Observed unique outcomes (choice):", x_o[:, -1].unique().tolist())
    return x_o.detach().cpu(), pulse_sides.detach().cpu()


def main():
    device = torch.device("cpu")
    print("Device:", device)

    # Determine pulse length P from constants/time discretization
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    # ---- component priors on device ----
    a0_prior  = Beta(torch.tensor(5.0, device=device), torch.tensor(5.0, device=device), validate_args=False)
    lam_prior = LogNormal(torch.tensor(0.0, device=device), torch.tensor(0.5, device=device), validate_args=False)
    v_prior   = LogNormal(torch.tensor(0.0, device=device), torch.tensor(0.5, device=device), validate_args=False)
    B_prior   = LogNormal(torch.tensor(0.5, device=device), torch.tensor(0.5, device=device), validate_args=False)

    tnd_prior = TransformedDistribution(
        base_distribution=Beta(torch.tensor(2.0, device=device), torch.tensor(5.0, device=device), validate_args=False),
        transforms=[AffineTransform(
            loc=torch.tensor(0.0, device=device),
            scale=torch.tensor(0.9, device=device),
        )],
        validate_args=False,
    )

    prior_theta = JointPrior(a0_prior, lam_prior, v_prior, B_prior, tnd_prior, device=device)

    # Proposal over pulse conditions (training only)
    pulse_prop = PulseSidePrior(P=P, p_success=0.75, seed=0, device=device)

    # Extended proposal over [theta, pulses]
    proposal = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device=device)

    # ---- simulate training set (theta + conditions) ----
    z_train, x_train = simulate_training_set_with_conditions(
        proposal,
        num_simulations=50_000,
        batch_size=4096,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
        P=P,
    )

    # ---- build & train MNLE ----
    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=True,   
        z_score_theta="independent",
        z_score_x="independent",
    )

    trainer = MNLE(prior=proposal, density_estimator=estimator_builder, device=str(device))
    trainer.append_simulations(z_train, x_train, data_device="cpu")
    estimator = trainer.train(training_batch_size=4096)
    print("Estimator device:", next(estimator.parameters()).device)

    # --- prior -- 
    theta_true = torch.tensor([0.55, 0.2, 1.0, 2.4, 0.25], device=device, dtype=torch.float32)

    num_trials = 300  # set to 3000 later if you want, but confirm behavior first.
    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=num_trials,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
        P=P,
        seed=123,
    )

    potential_fn = LikelihoodBasedPotential(
        estimator=estimator,
        proposal=proposal,
        x_o=x_o, 
    )

    # Condition on the observed pulses (must be shape (n_trials, P))
    conditioned_potential_fn = potential_fn.condition_on_theta(
        pulses_o,
        dims_global_theta=[0,1,2,3,4],
    )
    
    mcmc_kwargs = dict(
        num_chains=1,
        warmup_steps=1200,
        thin=1,
        init_strategy="resample",
        num_workers=1,
    )

    # Posterior over theta only: proposal must be *prior_theta* (not the extended proposal)
    prior_transform = mcmc_transform(prior_theta)

    mnle_posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        proposal=prior_theta,          # IMPORTANT: prior, not proposal
        theta_transform=prior_transform,
        **mcmc_kwargs,
    )

    samples = mnle_posterior.sample((10000,), x=x_o, show_progress_bars=True).detach().cpu()

    # ---- plot ----
    labels = [r"$a_0$", r"$\lambda$", r"$v$", r"$B$", r"$t_{nd}$"]
    fig, ax = pairplot(
        [prior_theta.sample((10000,)).detach().cpu(), samples],
        points=theta_true.detach().cpu().unsqueeze(0),
        diag="kde",
        upper="kde",
        labels=labels,
    )
    plt.show()


if __name__ == "__main__":
    main()
