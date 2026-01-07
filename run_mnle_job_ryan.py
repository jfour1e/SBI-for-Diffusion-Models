import os
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

import matplotlib.pyplot as plt

from torch.distributions import Beta, LogNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch.distributions import Distribution

from sbi.analysis import pairplot
from sbi.inference import MNLE, MCMCPosterior
from sbi.utils.get_nn_models import likelihood_nn
from sbi.utils import mcmc_transform

# IMPORTANT: uses your refactored simulator that accepts pulse_sides
from sbi_for_diffusion_models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    pulse_schedule,
    n_pulses_max_from_schedule,
    generate_pulse_matrix_numpy,
)

# ----------------------------
# Config
# ----------------------------

DEVICE = torch.device(os.environ.get("SBI_DEVICE", "cpu"))

# Data / simulator settings
MU_SENSORY = 1.0
P_SUCCESS = 0.75

# Training settings
NUM_SIMULATIONS = 1_000_000
TRAIN_BATCH_SIZE = 4096

# Observed-data settings
# Start small; likelihood approximation bias can explode when summing over many trials.
NUM_TRIALS_OBS = 200

# x preprocessing:
# We recommend log-transforming RT but NOT the categorical choice.
LOG_RT_MANUALLY = True

# If your sbi version supports log_transform_x for MNLE (log RT but not choice),
# you can set LOG_RT_MANUALLY=False and SBI_LOG_TRANSFORM_X=True.
SBI_LOG_TRANSFORM_X = False

# z-scoring of x inside the network. Often helps, especially for RT-like variables.
Z_SCORE_X = "independent"  # set to None to disable

# MCMC settings
NUM_CHAINS = 2
WARMUP_STEPS = 1200
POSTERIOR_SAMPLES = 10_000

# Chunk size over trials inside the potential (controls memory).
TRIAL_CHUNK_SIZE = 512

# Optional likelihood tempering for debugging only (1.0 = true posterior).
# If you see crazy posteriors at large NUM_TRIALS_OBS, try TEMPERATURE=10 or 100 to diagnose.
TEMPERATURE = 1.0

# Prior mode:
# - "narrow": your original priors (can make theta_true OOD if you set B~24 etc.)
# - "wide": covers regimes like small lam (~0.2) and large B (~24)
PRIOR_MODE = "wide"

# Whether to use a theta_true drawn from the prior (recommended for pipeline sanity checks).
THETA_TRUE_FROM_PRIOR = True

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
# Pulse-condition proposal for training
# (log_prob is constant because we only need sampling for training).
# ----------------------------

class PulseSequenceProposal(Distribution):
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
# x packing / preprocessing
# ----------------------------

def pack_x_rt_choice(rt_choice: torch.Tensor, *, log_rt: bool) -> torch.Tensor:
    """
    MNLE expects x to contain a continuous component(s) and then a discrete/categorical component
    in the last dimension. We keep choice values in {0,1,2} and store as float (common in sbi MNLE
    examples), but importantly we *do not* apply log to the choice.
    """
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)


# ----------------------------
# Simulator wrapper: (theta_and_pulses) -> x = [rt, choice]
# ----------------------------

@torch.no_grad()
def sim_wrapper(theta_and_pulses: torch.Tensor, *, mu_sensory: float, p_success: float, P: int, log_rt: bool):
    theta = theta_and_pulses[:, :5]
    pulse_sides = theta_and_pulses[:, 5:5 + P]

    rt_choice = rt_choice_model_simulator_torch(
        theta,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,  # not used if pulse_sides provided; safe
    )
    return pack_x_rt_choice(rt_choice, log_rt=log_rt)


# ----------------------------
# Training set generation: stream in batches to control memory
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


# ----------------------------
# Observed data generation (demo)
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


def summarize_trials(name: str, x: torch.Tensor):
    """
    x: (n_trials,2) where x[:,0] is (log)rt and x[:,1] is choice in {0,1,2}.
    """
    assert x.ndim == 2 and x.shape[1] == 2
    rt = x[:, 0]
    c = x[:, 1].to(torch.int64)

    counts = torch.bincount(c, minlength=3).cpu().numpy()
    frac = counts / max(counts.sum(), 1)

    print(
        f"{name}: n={len(x)}  "
        f"rt[min,max]=({rt.min().item():.4f},{rt.max().item():.4f})  "
        f"choice counts={counts.tolist()}  frac={frac.tolist()}"
    )


# ----------------------------
# Chunked conditioned potential: posterior(theta | x_o, pulses_o)
# ----------------------------

class ThetaConditionedOnPulsesPotential:
    """
    Potential function for posterior over theta given observed (x_i, pulses_i).

    We trained MNLE on z = [theta, pulses]. During inference, pulses are observed covariates
    that vary per trial. The likelihood is:
        p(x_o | theta, pulses_o) = ∏_i q_phi(x_i | theta, pulses_i)

    This class computes:
        log p(theta | x_o, pulses_o) ∝ log p(theta) + Σ_i log q_phi(x_i | theta, pulses_i)

    It computes Σ_i in chunks over i to control memory use.
    """

    def __init__(
        self,
        *,
        density_estimator,
        prior_theta: JointPrior,
        x_o: torch.Tensor,
        pulses_o: torch.Tensor,
        trial_chunk_size: int = 512,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.density_estimator = density_estimator.to(device)
        self.prior_theta = prior_theta
        self._x_o = x_o.to(device=device, dtype=torch.float32)
        self.pulses_o = pulses_o.to(device=device, dtype=torch.float32)

        self.trial_chunk_size = int(trial_chunk_size)
        self.temperature = float(temperature)
        self.device = device  # sbi reads potential_fn.device in some paths

        if self._x_o.ndim != 2 or self._x_o.shape[1] != 2:
            raise ValueError(f"x_o must be (n_trials,2), got {tuple(self._x_o.shape)}")
        if self.pulses_o.ndim != 2:
            raise ValueError(f"pulses_o must be (n_trials,P), got {tuple(self.pulses_o.shape)}")
        if self.pulses_o.shape[0] != self._x_o.shape[0]:
            raise ValueError(
                f"pulses_o and x_o must have same n_trials, got "
                f"{self.pulses_o.shape[0]} vs {self._x_o.shape[0]}"
            )

    # sbi helpers sometimes call these:
    def return_x_o(self):
        return self._x_o

    def set_x_o(self, x_o: torch.Tensor):
        self._x_o = x_o.to(self.device, dtype=torch.float32)
        return self

    def set_x(self, x: torch.Tensor):
        return self.set_x_o(x)

    def __call__(self, theta: torch.Tensor, track_gradients: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(bool(track_gradients)):
            if theta.ndim == 1:
                theta = theta.view(1, -1)
            if theta.shape[-1] != 5:
                raise ValueError(f"Expected theta shape (N,5), got {tuple(theta.shape)}")

            theta = theta.to(self.device, dtype=torch.float32)

            # Prior term
            lp = self.prior_theta.log_prob(theta)  # (N,)

            valid = torch.isfinite(lp)
            if not torch.any(valid):
                return lp

            theta_v = theta[valid]  # (Nv,5)
            ll_v = self._log_likelihood_theta(theta_v)  # (Nv,)

            out = lp.clone()
            out[valid] = out[valid] + ll_v
            return out

    def _log_likelihood_theta(self, theta_v: torch.Tensor) -> torch.Tensor:
        Nv = theta_v.shape[0]
        n_trials = self._x_o.shape[0]
        P = self.pulses_o.shape[1]

        ll = torch.zeros((Nv,), dtype=torch.float32, device=self.device)

        for t0 in range(0, n_trials, self.trial_chunk_size):
            t1 = min(n_trials, t0 + self.trial_chunk_size)
            tb = t1 - t0

            x_b = self._x_o[t0:t1]            # (tb,2)
            pulses_b = self.pulses_o[t0:t1]   # (tb,P)

            theta_rep = theta_v[:, None, :].expand(Nv, tb, 5)
            pulses_rep = pulses_b[None, :, :].expand(Nv, tb, P)

            z = torch.cat([theta_rep, pulses_rep], dim=-1).reshape(Nv * tb, 5 + P)
            x = x_b[None, :, :].expand(Nv, tb, 2).reshape(Nv * tb, 2)

            ll += self.density_estimator.log_prob(x, z).view(Nv, tb).sum(dim=1)

        if self.temperature != 1.0:
            ll = ll / self.temperature

        return ll

# --- NEW: helper to simulate one SBC dataset ---
@torch.no_grad()
def simulate_one_sbc_dataset(
    theta_true: torch.Tensor,
    *,
    num_trials: int,
    device,
    mu_sensory: float,
    p_success: float,
    P: int,
    seed: int,
    log_rt: bool,
):
    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=num_trials,
        device=device,
        mu_sensory=mu_sensory,
        p_success=p_success,
        P=P,
        seed=seed,
        log_rt=log_rt,
    )
    return x_o, pulses_o

def sbc_manual(
    *,
    prior_theta,
    density_estimator,
    P: int,
    simulate_observed_session_fn,
    mcmc_transform_fn,
    num_sbc_runs: int = 200,
    num_trials: int = 50,
    num_posterior_samples: int = 1000,
    mu_sensory: float = 1.0,
    p_success: float = 0.75,
    log_rt: bool = False,
    trial_chunk_size: int = 512,
    temperature: float = 1.0,
    warmup_steps: int = 600,
    num_chains: int = 1,
    device_sim=None,   # DEVICE from your script
    seed0: int = 12345,
):
    """
    Manual SBC:
      For i=1..N:
        theta_i ~ prior
        (x_i, pulses_i) ~ simulator(theta_i)
        draw posterior samples theta~p_hat(theta|x_i,pulses_i)
        rank_i = #{posterior_samples < theta_i} per dim
    Returns ranks: (N, dim)
    """
    dim = 5
    ranks = torch.empty((num_sbc_runs, dim), dtype=torch.int64)

    theta_transform = mcmc_transform_fn(prior_theta)

    for i in range(num_sbc_runs):
        # 1) sample theta from prior
        theta_true = prior_theta.sample((1,)).view(dim).to(torch.float32)

        # 2) simulate dataset
        x_o, pulses_o = simulate_observed_session_fn(
            theta_true,
            num_trials=num_trials,
            device=device_sim,
            mu_sensory=mu_sensory,
            p_success=p_success,
            P=P,
            seed=seed0 + i,
            log_rt=log_rt,
        )

        # 3) build potential + posterior (your working method)
        potential = ThetaConditionedOnPulsesPotential(
            density_estimator=density_estimator,
            prior_theta=prior_theta,
            x_o=x_o,
            pulses_o=pulses_o,
            trial_chunk_size=trial_chunk_size,
            temperature=temperature,
            device="cpu",
        )

        posterior = MCMCPosterior(
            potential_fn=potential,
            proposal=prior_theta,
            theta_transform=theta_transform,
            num_chains=num_chains,
            warmup_steps=warmup_steps,
            thin=1,
            init_strategy="prior",
            num_workers=1,
        )

        post_samps = posterior.sample((num_posterior_samples,), show_progress_bars=False).detach().cpu()

        # 4) rank statistic per dimension
        # rank = number of posterior draws less than true value
        theta_true_cpu = theta_true.detach().cpu()
        ranks[i] = (post_samps < theta_true_cpu[None, :]).sum(dim=0)

        if (i + 1) % max(1, num_sbc_runs // 10) == 0:
            print(f"SBC {i+1}/{num_sbc_runs}")

    return ranks

def plot_sbc_ranks(ranks: torch.Tensor, num_posterior_samples: int, num_bins: int = 20):
    """
    ranks: (N, dim), each entry in {0,...,num_posterior_samples}
    Uniform histogram per dim indicates calibration.
    """
    ranks = ranks.numpy()
    N, dim = ranks.shape
    fig, axes = plt.subplots(1, dim, figsize=(3.2 * dim, 3), sharey=True)
    if dim == 1:
        axes = [axes]
    bins = np.linspace(0, num_posterior_samples, num_bins + 1)

    for d in range(dim):
        axes[d].hist(ranks[:, d], bins=bins, edgecolor="black")
        axes[d].set_title(f"param {d} rank")
        axes[d].set_xlabel("rank")
    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.show()


def main():
    print("Device:", DEVICE)

    # Determine pulse length P from constants/time discretization
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    # ----------------------------
    # Priors
    # ----------------------------
    a0_prior = Beta(torch.tensor(10.0, device=DEVICE), torch.tensor(10.0, device=DEVICE), validate_args=False)

    if PRIOR_MODE == "wide":
        # Wide enough to include lam ~ 0.2 and B ~ 24 as non-extreme events.
        lam_prior = LogNormal(torch.tensor(-1.0, device=DEVICE), torch.tensor(1.0, device=DEVICE), validate_args=False)
        B_prior   = LogNormal(torch.tensor(2.75, device=DEVICE), torch.tensor(0.5, device=DEVICE), validate_args=False)
    else:
        # Your original settings (may be too narrow for large B / very small lam).
        lam_prior = LogNormal(torch.tensor(0.0, device=DEVICE), torch.tensor(0.5, device=DEVICE), validate_args=False)
        B_prior   = LogNormal(torch.tensor(0.3, device=DEVICE), torch.tensor(0.5, device=DEVICE), validate_args=False)

    v_prior = LogNormal(torch.tensor(0.0, device=DEVICE), torch.tensor(0.5, device=DEVICE), validate_args=False)

    tnd_prior = TransformedDistribution(
        base_distribution=Beta(torch.tensor(2.0, device=DEVICE), torch.tensor(5.0, device=DEVICE), validate_args=False),
        transforms=[AffineTransform(
            loc=torch.tensor(0.0, device=DEVICE),
            scale=torch.tensor(0.9, device=DEVICE),
        )],
        validate_args=False,
    )

    prior_theta = JointPrior(a0_prior, lam_prior, v_prior, B_prior, tnd_prior, device=DEVICE)

    # Proposal over pulse conditions (training only)
    pulse_prop = PulseSequenceProposal(P=P, p_success=P_SUCCESS, seed=0, device=DEVICE)

    # Extended proposal over [theta, pulses]
    proposal = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device=DEVICE)

    # ----------------------------
    # Simulate training set
    # ----------------------------
    print("\n--- Simulating training set ---")
    z_train, x_train = simulate_training_set_with_conditions(
        proposal,
        num_simulations=NUM_SIMULATIONS,
        batch_size=TRAIN_BATCH_SIZE,
        device=DEVICE,
        mu_sensory=MU_SENSORY,
        p_success=P_SUCCESS,
        P=P,
        log_rt=LOG_RT_MANUALLY,
    )

    summarize_trials("train (sample)", x_train[torch.randperm(len(x_train))[:50_000]])

    # ----------------------------
    # Build & train MNLE
    # ----------------------------
    print("\n--- Training MNLE ---")
    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=bool(SBI_LOG_TRANSFORM_X),
        z_score_theta="independent",
        z_score_x=Z_SCORE_X,
    )

    trainer = MNLE(prior=proposal, density_estimator=estimator_builder, device=str(DEVICE))
    trainer = trainer.append_simulations(z_train, x_train, data_device="cpu")
    density_estimator = trainer.train(training_batch_size=TRAIN_BATCH_SIZE)
    print("Estimator device:", next(density_estimator.parameters()).device)

    # ----------------------------
    # Observed data (demo)
    # ----------------------------
    print("\n--- Simulating observed session ---")
    if THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5).to(torch.float32)
    else:
        # If you set THETA_TRUE_FROM_PRIOR=False, ensure PRIOR_MODE is wide enough for this.
        theta_true = torch.tensor([0.55, 0.2, 1.0, 24.0, 0.25], device=DEVICE, dtype=torch.float32)

    print("theta_true =", theta_true.detach().cpu().numpy().tolist())

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=NUM_TRIALS_OBS,
        device=DEVICE,
        mu_sensory=MU_SENSORY,
        p_success=P_SUCCESS,
        P=P,
        seed=123,
        log_rt=LOG_RT_MANUALLY,
    )

    summarize_trials("obs", x_o)

    # ----------------------------
    # MCMC posterior over theta only
    # ----------------------------
    print("\n--- MCMC posterior ---")
    potential_fn = ThetaConditionedOnPulsesPotential(
        density_estimator=density_estimator,
        prior_theta=prior_theta,
        x_o=x_o,
        pulses_o=pulses_o,
        trial_chunk_size=TRIAL_CHUNK_SIZE,
        temperature=TEMPERATURE,
        device="cpu",  # keep on CPU for broad compatibility
    )

    # Transform theta to unconstrained space (may or may not help depending on your sbi version / prior).
    theta_transform = mcmc_transform(prior_theta)

    posterior = MCMCPosterior(
        potential_fn=potential_fn,
        proposal=prior_theta,
        theta_transform=theta_transform,
        num_chains=NUM_CHAINS,
        warmup_steps=WARMUP_STEPS,
        thin=1,
        init_strategy="prior",
        num_workers=1,
    )

    # Prefer slice sampling for robustness if available (API varies across sbi versions).
    if hasattr(posterior, "set_mcmc_method"):
        try:
            posterior = posterior.set_mcmc_method("slice_np")
            print("Using MCMC method: slice_np")
        except Exception as e:
            print("Could not set mcmc_method to slice_np; using default. Error:", repr(e))

    samples = posterior.sample((POSTERIOR_SAMPLES,), show_progress_bars=True).detach().cpu()

    # ----------------------------
    # Plot
    # ----------------------------
    labels = [r"$a_0$", r"$\lambda$", r"$v$", r"$B$", r"$t_{nd}$"]
    fig, ax = pairplot(
        [prior_theta.sample((10_000,)).detach().cpu(), samples],
        points=theta_true.detach().cpu().unsqueeze(0),
        diag="kde",
        upper="kde",
        labels=labels,
    )
    plt.show()

    # ----------------------------
    # NEW: Simulation-Based Calibration (SBC)
    # ----------------------------

    print("\n--- Manual SBC ---")
    ranks = sbc_manual(
        prior_theta=prior_theta,
        density_estimator=density_estimator,
        P=P,
        simulate_observed_session_fn=simulate_observed_session,
        mcmc_transform_fn=mcmc_transform,
        num_sbc_runs=200,           
        num_trials=50,
        num_posterior_samples=500,
        mu_sensory=MU_SENSORY,
        p_success=P_SUCCESS,
        log_rt=LOG_RT_MANUALLY,
        trial_chunk_size=TRIAL_CHUNK_SIZE,
        temperature=TEMPERATURE,
        warmup_steps=400,
        num_chains=1,
        device_sim=DEVICE,
    )
    plot_sbc_ranks(ranks, num_posterior_samples=500, num_bins=20)

    print("\nDone.")
    print("Next step: gradually increase NUM_TRIALS_OBS (e.g., 200 -> 500 -> 1000).")
    print("If posterior becomes unstable as trials increase, that strongly suggests MNLE bias accumulation.")

if __name__ == "__main__":
    main()
