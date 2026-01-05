import torch
torch.distributions.Distribution.set_default_validate_args(False)
import matplotlib.pyplot as plt
import numpy as np

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
# We do NOT need an exact log_prob; this is a proposal used for MNLE training.
# ----------------------------

class PulseSequenceProposal(Distribution):
    """
    Samples pulse_sides in {-1,+1}^{P} using your existing generator.
    log_prob is set to 0 (constant) because we only need sampling for training.
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
        # sample_shape may be multi-dim; we generate prod(sample_shape) sequences and reshape
        n = int(np.prod(sample_shape)) if len(sample_shape) > 0 else 1
        s_np = generate_pulse_matrix_numpy(self.rng, n_trials=n, n_pulses=self.P, p_success=self.p_success)
        s = torch.from_numpy(s_np).to(dtype=torch.float32)
        if len(sample_shape) > 0:
            s = s.view(*sample_shape, self.P)
        if self._device is not None:
            s = s.to(self._device)
        return s

    def log_prob(self, value):
        # Constant log-prob is OK for a training proposal.
        return torch.zeros(value.shape[:-1], device=value.device, dtype=torch.float32)


class ExtendedProposal(Distribution):
    """
    Proposal over concatenated [theta(5), pulse_sides(P)] used to train MNLE.
    """
    arg_constraints = {}
    has_rsample = False

    def __init__(self, theta_prior: JointPrior, pulse_proposal: PulseSequenceProposal, device=None):
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
# Pack x for MNLE: x = [rt, choice], with choice LAST (categorical).
# ----------------------------

def pack_x_rt_choice(rt_choice: torch.Tensor) -> torch.Tensor:
    """
    rt_choice: (N,2) with columns [rt, choice] where choice in {0,1,2}.
    returns x: (N,2) float32 with choice last (still exact integers as floats).
    """
    if rt_choice.ndim != 2 or rt_choice.shape[1] != 2:
        raise ValueError(f"rt_choice must be (N,2), got {tuple(rt_choice.shape)}")

    rt = rt_choice[:, 0:1].to(torch.float32)
    choice_int = rt_choice[:, 1].to(torch.int64)

    if not torch.all((choice_int == 0) | (choice_int == 1) | (choice_int == 2)):
        raise ValueError(f"choice must be in {{0,1,2}}; found {torch.unique(choice_int).tolist()}")

    x = torch.cat([rt, choice_int.view(-1, 1).to(torch.float32)], dim=1)
    return x


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
    z_all = proposal.sample((num_simulations,)).to(device=device, dtype=torch.float32)

    xs = []
    for start in range(0, num_simulations, batch_size):
        z = z_all[start:start + batch_size]
        x = sim_wrapper(z, mu_sensory=mu_sensory, p_success=p_success, P=P)
        xs.append(x.detach().cpu())

    x_all = torch.cat(xs, dim=0).to(torch.float32)
    z_cpu = z_all.detach().cpu()

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
    B_prior   = LogNormal(torch.tensor(0.3, device=device), torch.tensor(0.5, device=device), validate_args=False)

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
    pulse_prop = PulseSequenceProposal(P=P, p_success=0.75, seed=0, device=device)

    # Extended proposal over [theta, pulses]
    proposal = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device=device)

    # ---- simulate training set (theta + conditions) ----
    z_train, x_train = simulate_training_set_with_conditions(
        proposal,
        num_simulations=500_000,
        batch_size=4096,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
        P=P,
    )

    # ---- build & train MNLE ----
    # Now x = [rt, choice], so log_transform_x=True is appropriate and recommended.
    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=True,
        z_score_theta="independent",
        z_score_x=None,  # keep discrete column exact
    )

    trainer = MNLE(prior=proposal, density_estimator=estimator_builder, device=str(device))
    trainer = trainer.append_simulations(z_train, x_train, data_device="cpu")
    density_estimator = trainer.train(training_batch_size=4096)
    print("Estimator device:", next(density_estimator.parameters()).device)

    # ---- observed data ----
    theta_true = torch.tensor([0.55, 0.2, 1.5, 2.0, 0.25], device=device, dtype=torch.float32)

    # NOTE: MNLE bias can accumulate over thousands of trials; start smaller and scale up carefully.
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

    class ConditionedPulsePotential:
        """
        Potential for posterior over global theta given observed (x_i, pulses_i).
        Uses the trained MNLE estimator q_phi(x | z) with z = [theta, pulses_i].
        """

        def __init__(self, density_estimator, prior_theta, x_o, pulses_o):
            self.density_estimator = density_estimator.to("cpu")
            self.prior_theta = prior_theta
            self._x_o = x_o.to("cpu", dtype=torch.float32)            # store as _x_o
            self.pulses_o = pulses_o.to("cpu", dtype=torch.float32)
            self.device = "cpu"  # IMPORTANT: sbi reads potential_fn.device

            if self._x_o.ndim != 2 or self._x_o.shape[1] != 2:
                raise ValueError(f"x_o must be (n_trials,2), got {tuple(self._x_o.shape)}")
            if self.pulses_o.ndim != 2:
                raise ValueError(f"pulses_o must be (n_trials,P), got {tuple(self.pulses_o.shape)}")
            if self.pulses_o.shape[0] != self._x_o.shape[0]:
                raise ValueError(
                    f"pulses_o and x_o must have same n_trials, got "
                    f"{self.pulses_o.shape[0]} vs {self._x_o.shape[0]}"
                )

        # --- REQUIRED by your sbi BasePosterior ---
        def return_x_o(self):
            return self._x_o

        # (Optional but often expected in some sbi utilities; safe to include.)
        def set_x_o(self, x_o: torch.Tensor):
            self._x_o = x_o.to("cpu", dtype=torch.float32)
            return self
        
        def set_x(self, x: torch.Tensor):
            return self.set_x_o(x)

    class ConditionedPulsePotential:
        """
        Potential for posterior over global theta given observed (x_i, pulses_i).
        Uses the trained MNLE estimator q_phi(x | z) with z = [theta, pulses_i].
        """

        def __init__(self, density_estimator, prior_theta, x_o, pulses_o):
            self.density_estimator = density_estimator.to("cpu")
            self.prior_theta = prior_theta
            self._x_o = x_o.to("cpu", dtype=torch.float32)            # store as _x_o
            self.pulses_o = pulses_o.to("cpu", dtype=torch.float32)
            self.device = "cpu"  # IMPORTANT: sbi reads potential_fn.device

            if self._x_o.ndim != 2 or self._x_o.shape[1] != 2:
                raise ValueError(f"x_o must be (n_trials,2), got {tuple(self._x_o.shape)}")
            if self.pulses_o.ndim != 2:
                raise ValueError(f"pulses_o must be (n_trials,P), got {tuple(self.pulses_o.shape)}")
            if self.pulses_o.shape[0] != self._x_o.shape[0]:
                raise ValueError(
                    f"pulses_o and x_o must have same n_trials, got "
                    f"{self.pulses_o.shape[0]} vs {self._x_o.shape[0]}"
                )

        # --- REQUIRED by your sbi BasePosterior ---
        def return_x_o(self):
            return self._x_o

        # (Optional but often expected in some sbi utilities; safe to include.)
        def set_x_o(self, x_o: torch.Tensor):
            self._x_o = x_o.to("cpu", dtype=torch.float32)
            return self
        
        def set_x(self, x: torch.Tensor):
            return self.set_x_o(x)

        def __call__(self, theta: torch.Tensor, track_gradients: bool = False) -> torch.Tensor:
            # sbi may pass track_gradients even when using non-gradient samplers
            with torch.set_grad_enabled(bool(track_gradients)):
                if theta.ndim == 1:
                    theta = theta.view(1, -1)
                if theta.shape[-1] != 5:
                    raise ValueError(f"Expected theta shape (N,5), got {tuple(theta.shape)}")

                theta = theta.to("cpu", dtype=torch.float32)

                lp = self.prior_theta.log_prob(theta)  # (N,)
                valid = torch.isfinite(lp)
                if not torch.any(valid):
                    return lp

                theta_valid = theta[valid]
                Nv = theta_valid.shape[0]
                n_trials = self._x_o.shape[0]
                P = self.pulses_o.shape[1]

                theta_rep = theta_valid[:, None, :].expand(Nv, n_trials, 5)
                pulses_rep = self.pulses_o[None, :, :].expand(Nv, n_trials, P)

                z = torch.cat([theta_rep, pulses_rep], dim=-1).reshape(Nv * n_trials, 5 + P)
                x = self._x_o[None, :, :].expand(Nv, n_trials, 2).reshape(Nv * n_trials, 2)

                ll = self.density_estimator.log_prob(x, z).view(Nv, n_trials).sum(dim=1)

                out = lp.clone()
                out[valid] = out[valid] + ll
                return out


    # Create the conditioned potential over theta only
    conditioned_potential_fn = ConditionedPulsePotential(
        density_estimator=density_estimator,
        prior_theta=prior_theta,
        x_o=x_o,
        pulses_o=pulses_o,
    )


    # ---- MCMC posterior over theta only ----
    prior_transform = mcmc_transform(prior_theta)

    mcmc_kwargs = dict(
        num_chains=1,
        warmup_steps=1200,
        thin=1,
        init_strategy="resample",
        num_workers=1,
    )

    posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        proposal=prior_theta,
        theta_transform=prior_transform,
        **mcmc_kwargs,
    )

    samples = posterior.sample((300,), show_progress_bars=True).detach().cpu()

    # ---- plot ----
    labels = [r"$a_0$", r"$\lambda$", r"$v$", r"$B$", r"$t_{nd}$"]
    fig, ax = pairplot(
        [prior_theta.sample((2000,)).detach().cpu(), samples],
        points=theta_true.detach().cpu().unsqueeze(0),
        diag="kde",
        upper="kde",
        labels=labels,
    )
    plt.show()


if __name__ == "__main__":
    main()
