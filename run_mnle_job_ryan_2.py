import os
import numpy as np
import torch

# Avoid expensive distribution argument checks in tight loops.
torch.distributions.Distribution.set_default_validate_args(False)

import matplotlib.pyplot as plt

from torch.distributions import Beta, LogNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch.distributions import Distribution

from sbi.analysis import pairplot
from sbi.inference import MNLE, MCMCPosterior
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.neural_nets import likelihood_nn
from sbi.utils import MultipleIndependent, mcmc_transform

from sbi_for_diffusion_models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    pulse_schedule,
    n_pulses_max_from_schedule,
    generate_pulse_matrix_numpy,
)

"""
Configuration settings
"""
# Data / simulator settings
MU_SENSORY = 1.0
P_SUCCESS = 0.75

# Training settings
NUM_SIMULATIONS = 1_000_000
TRAIN_BATCH_SIZE = 4096

# Observed-data settings
# Start small; likelihood approximation bias can explode when summing over many trials.
NUM_TRIALS_OBS = 50

# x preprocessing:
# We recommend log-transforming RT but NOT the categorical choice.
LOG_RT_MANUALLY = False

# If your sbi version supports log_transform_x for MNLE (log RT but not choice),
# you can set LOG_RT_MANUALLY=False and SBI_LOG_TRANSFORM_X=True.
SBI_LOG_TRANSFORM_X = True

# z-scoring of x inside the network. Often helps, especially for RT-like variables.
Z_SCORE_X = "independent"  # set to None to disable

# MCMC settings
NUM_CHAINS = 2
WARMUP_STEPS = 1200
POSTERIOR_SAMPLES = 10_000

# Optional likelihood tempering for debugging only (1.0 = true posterior).
# If you see crazy posteriors at large NUM_TRIALS_OBS, try TEMPERATURE=10 or 100 to diagnose.
TEMPERATURE = 1.0

# Whether to use a theta_true drawn from the prior (recommended for pipeline sanity checks).
THETA_TRUE_FROM_PRIOR = True


# helper functions
def pack_x_rt_choice(rt_choice: torch.Tensor, *, log_rt: bool) -> torch.Tensor:
    """
    MNLE expects x to contain continuous component(s) and then a discrete/categorical
    component in the last dimension. We keep choice values in {0,1,2} and store as float,
    but we do *not* apply log to choice.
    """
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)


def sim_wrapper(
    theta_and_pulses: torch.Tensor, *, mu_sensory: float, p_success: float, P: int, log_rt: bool
) -> torch.Tensor:
    """
    Simulator wrapper that expects concatenated z = [theta(5), pulse_sides(P)].
    Returns packed x = [rt(or log rt), choice].
    """
    theta = theta_and_pulses[:, :5]
    pulse_sides = theta_and_pulses[:, 5 : 5 + P]

    rt_choice = rt_choice_model_simulator_torch(
        theta,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,  # not used if pulse_sides provided; safe
    )
    return pack_x_rt_choice(rt_choice, log_rt=log_rt)


@torch.no_grad()
def simulate_training_set_with_conditions(
    proposal: Distribution,
    num_simulations: int,
    batch_size: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
    log_rt: bool,
):
    zs = []
    xs = []

    for start in range(0, num_simulations, batch_size):
        bs = min(batch_size, num_simulations - start)
        z = proposal.sample((bs,)).to(device=device, dtype=torch.float32)
        x = sim_wrapper(z, mu_sensory=mu_sensory, p_success=p_success, P=P, log_rt=log_rt)

        zs.append(z.detach().cpu())
        xs.append(x.detach().cpu())

        if (start // batch_size) % 50 == 0:
            print(f"Simulated {start + bs:,}/{num_simulations:,}")

    z_all = torch.cat(zs, dim=0).to(torch.float32)
    x_all = torch.cat(xs, dim=0).to(torch.float32)

    assert z_all.shape[0] == num_simulations
    assert x_all.shape[0] == num_simulations
    assert torch.isfinite(z_all).all()
    assert torch.isfinite(x_all).all()
    assert torch.all((x_all[:, -1] == 0) | (x_all[:, -1] == 1) | (x_all[:, -1] == 2))

    print("Training x shape:", tuple(x_all.shape), " (N,2) = [rt(or log rt), choice]")
    print("Training z shape:", tuple(z_all.shape), " (N, 5+P) = [theta, pulses]")
    print("Unique outcomes in training (choice):", x_all[:, -1].unique().tolist())
    return z_all, x_all


def summarize_trials(name: str, x: torch.Tensor) -> None:
    rt = x[:, 0]
    choice = x[:, 1].to(torch.int64)
    counts = torch.bincount(choice, minlength=3)
    frac = counts.float() / counts.sum().clamp_min(1)
    print(
        f"{name}: n={len(x)}  "
        f"rt[min,max]=({rt.min().item():.4f},{rt.max().item():.4f})  "
        f"choice counts={counts.tolist()}  frac={frac.tolist()}"
    )


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
    log_rt: bool,
):
    rng = np.random.default_rng(seed)
    s_np = generate_pulse_matrix_numpy(rng, n_trials=num_trials, n_pulses=P, p_success=p_success)
    pulses_o = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

    theta_rep = theta_true.view(1, 5).repeat(num_trials, 1)
    rt_choice = rt_choice_model_simulator_torch(
        theta_rep,
        mu_sensory=mu_sensory,
        pulse_sides=pulses_o,
        p_success=p_success,
    )
    x_o = pack_x_rt_choice(rt_choice, log_rt=log_rt)

    return x_o.detach().cpu(), pulses_o.detach().cpu()


# TRAININ PROPOSAL
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


def main():
    # Deterministic-ish behavior for debugging.
    torch.manual_seed(0)
    np.random.seed(0)

    # Determine pulse length P from constants/time discretization
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    # priors
    prior_theta = MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),  # a0
            LogNormal(torch.tensor([-1.0]), torch.tensor([1.0])),  # lam
            LogNormal(torch.tensor([0.0]), torch.tensor([1.0])),  # v
            LogNormal(torch.tensor([2.75]), torch.tensor([0.5])),  # B
            Beta(torch.tensor([2.0]), torch.tensor([2.0])
            ),  # p_lapse in [0,0.9]
        ]
    )

                #TransformedDistribution(
                #base_distribution=Beta(torch.tensor([2.0]), torch.tensor([5.0])),
                #transforms=[AffineTransform(loc=torch.tensor([0.0]), scale=torch.tensor([0.9]))]

    # Training proposal over z=[theta,pulses]
    pulse_prop = PulseSequenceProposal(P=P, p_success=P_SUCCESS, seed=0, device="cpu")
    proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

    # simualte training set
    print("\n--- Simulating training set ---")
    z_train, x_train = simulate_training_set_with_conditions(
        proposal=proposal_z,
        num_simulations=NUM_SIMULATIONS,
        batch_size=TRAIN_BATCH_SIZE,
        device="cpu",
        mu_sensory=MU_SENSORY,
        p_success=P_SUCCESS,
        P=P,
        log_rt=LOG_RT_MANUALLY,
    )

    summarize_trials("train (sample)", x_train[torch.randperm(len(x_train))[:50_000]])

   # train mnle
    est_build = likelihood_nn(
        model="mnle",
        log_transform_x=bool(SBI_LOG_TRANSFORM_X),
        z_score_theta="independent",
        z_score_x=Z_SCORE_X,
        hidden_features=128,
        num_transforms=10,
        num_bins=24,
    )

    trainer = MNLE(prior=proposal_z, density_estimator=est_build, device="cpu")
    trainer = trainer.append_simulations(z_train, x_train)

    # sbi changed the argument name across versions.
    try:
        density_estimator = trainer.train(training_batch_size=TRAIN_BATCH_SIZE)
    except TypeError:
        density_estimator = trainer.train(batch_size=TRAIN_BATCH_SIZE)

# Simulate Known data
    print("\n--- Simulating observed session ---")
    if THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=MU_SENSORY,
        p_success=P_SUCCESS,
        P=P,
        seed=123,
        log_rt=LOG_RT_MANUALLY,
    )

    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # Conditioning
    base_potential = LikelihoodBasedPotential(density_estimator, proposal_z, x_o=x_o, device="cpu")

    # We are sampling only the 5 global theta parameters.
    dims_global_theta = list(range(5))

    # Condition on the known pulse sequence for each trial.
    conditioned_loglike = ConditionedMNLELogLikelihood(
        estimator=density_estimator,
        local_theta=pulses_o,   # (num_trials, P)
        device="cpu",
    )

    # Add log prior(theta) manually. condition_on_theta returns likelihood only.
    potential_theta = ThetaOnlyPosteriorPotential(
        conditioned_loglike=conditioned_loglike,
        prior_theta=prior_theta,
        x_o=x_o,
        device="cpu",
        temperature=TEMPERATURE,
    )

    theta_transform = mcmc_transform(prior_theta)

    posterior = MCMCPosterior(
        potential_fn=potential_theta,
        proposal=prior_theta,
        theta_transform=theta_transform,
        method="nuts_pyro",
        num_chains=NUM_CHAINS,
        warmup_steps=WARMUP_STEPS,
        thin=1,
        init_strategy="proposal",
        num_workers=1,
    )

    print("\n--- Sampling posterior over theta ---")
    samples = (posterior.sample((POSTERIOR_SAMPLES,), x=x_o, show_progress_bars=True).detach().cpu())


    # Plots and diagnostics
    outdir = os.environ.get("OUTDIR", "mnle_outputs")
    os.makedirs(outdir, exist_ok=True)

    fig, ax = pairplot(
        samples,
        points=theta_true.view(1, -1).cpu(),
        labels=["a0", "lam", "v", "B", "tau"],
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)

    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)


if __name__ == "__main__":
    main()
