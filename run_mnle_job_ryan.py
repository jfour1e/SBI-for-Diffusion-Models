import torch
import matplotlib.pyplot as plt
import numpy as np

from sbi.inference import MNLE
from sbi.analysis import pairplot
from sbi.utils.get_nn_models import likelihood_nn

from torch.distributions import Normal, Beta, LogNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch.distributions import Distribution

# IMPORTANT: you need the refactored simulator that accepts pulse_sides
from sbi_for_diffusion_models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    simulate_session_data_rt_choice,
    pulse_schedule,
    n_pulses_max_from_schedule,
    generate_pulse_matrix_numpy,
)

# ----------------------------
# Prior
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
# Data packing: x = [rt, choice, s1..sP]
# ----------------------------

def pack_x_for_mnle(rt_choice: torch.Tensor, pulse_sides: torch.Tensor) -> torch.Tensor:
    """
    Pack x for MNLE with the discrete variable LAST.

    Inputs
    ------
    rt_choice: (N,2) tensor with columns [rt, choice]
              choice must be in {0,1,2} (2 = censored)
    pulse_sides: (N,P) tensor with entries in {-1,+1}

    Output
    ------
    x: (N, 1+P+1) tensor with columns [rt, pulses..., choice]
       MNLE will treat last column as categorical.
    """
    if rt_choice.ndim != 2 or rt_choice.shape[1] != 2:
        raise ValueError(f"rt_choice must be (N,2), got {tuple(rt_choice.shape)}")
    if pulse_sides.ndim != 2:
        raise ValueError(f"pulse_sides must be (N,P), got {tuple(pulse_sides.shape)}")
    if pulse_sides.shape[0] != rt_choice.shape[0]:
        raise ValueError("rt_choice and pulse_sides must have same batch size N")

    rt = rt_choice[:, 0:1].to(torch.float32)

    # Ensure discrete labels are EXACT integers 0/1/2.
    # (MNLE will internally feed these into a Categorical.log_prob.)
    choice = rt_choice[:, 1].to(torch.int64)
    if not torch.all((choice == 0) | (choice == 1) | (choice == 2)):
        raise ValueError(
            f"choice must be in {{0,1,2}}; found {torch.unique(choice).tolist()}"
        )

    s = pulse_sides.to(torch.float32)

    # IMPORTANT: discrete variable must be last column
    x = torch.cat([rt, s, choice.view(-1, 1).to(torch.float32)], dim=1)
    return x


# ----------------------------
# Simulator wrapper for SBI: theta -> x_aug
# ----------------------------

@torch.no_grad()
def simulate_training_set_conditioned(
    prior,
    num_simulations: int,
    batch_size: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    seed: int = 0,
):
    """
    Generates training pairs (theta, x_aug) with x_aug including realized stimulus.
    This trains MNLE on p(rt,choice | theta, stimulus) because stimulus is included in x.
    """
    # theta on device
    theta = prior.sample((num_simulations,)).to(dtype=torch.float32)

    # Determine pulse length P from constants/time discretization
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

    # Use a NumPy RNG for pulse generation (matches your existing pulse logic)
    rng = np.random.default_rng(seed)

    xs = []
    for start in range(0, num_simulations, batch_size):
        th = theta[start:start + batch_size]
        n_b = th.shape[0]

        # Generate realized stimulus externally (conditioning variable)
        s_np = generate_pulse_matrix_numpy(rng, n_trials=n_b, n_pulses=P, p_success=p_success)
        s = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

        # Simulate rt/choice *conditioned on s*
        rt_choice = rt_choice_model_simulator_torch(
            th,
            mu_sensory=mu_sensory,
            pulse_sides=s,
            p_success=p_success,  # not used when pulse_sides is provided, but safe
        )

        x_aug = pack_x_for_mnle(rt_choice, s)
        xs.append(x_aug.detach().cpu())

    x = torch.cat(xs, dim=0).to(torch.float32)
    theta_cpu = theta.detach().cpu()

    assert torch.isfinite(theta_cpu).all()
    assert torch.isfinite(x).all()

    # choice column is x[:, -1]
    assert torch.all((x[:, -1] == 0) | (x[:, -1] == 1) | (x[:, -1] == 2))
    print("Unique outcomes in training (choice):", x[:, -1].unique().tolist())
    print("x_aug dim =", x.shape[1], "(= 2 + P)")
    return theta_cpu, x, P


# ----------------------------
# Observed session: produce x_o augmented with stimulus
# ----------------------------

@torch.no_grad()
def simulate_observed_session_conditioned(
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
    s = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

    # Use your session helper; it now supports pulse_sides and can return it too.
    # If your refactor includes return_pulse_sides, you can use that. Otherwise do it manually as below.
    rt_choice = simulate_session_data_rt_choice(
        theta_true,
        num_trials=num_trials,
        mu_sensory=mu_sensory,
        pulse_sides=s,
        p_success=p_success,
    )

    x_o = pack_x_for_mnle(rt_choice, s)
    print("Observed unique outcomes (choice):", x_o[:, -1].unique().tolist())
    return x_o


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- component priors on device ----
    a0_prior  = Beta(torch.tensor(5.0, device=device), torch.tensor(5.0, device=device), validate_args=False)
    lam_prior = Normal(torch.tensor(0.0, device=device), torch.tensor(0.5, device=device), validate_args=False)
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

    prior = JointPrior(a0_prior, lam_prior, v_prior, B_prior, tnd_prior, device=device)

    # ---- simulate training set (CONDITIONED) ----
    theta_train, x_train, P = simulate_training_set_conditioned(
        prior,
        num_simulations=50_000,
        batch_size=4096,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
        seed=0,
    )

    # ---- build & train MNLE ----
    # IMPORTANT:
    # - log_transform_x=True is NOT appropriate anymore because x contains negative pulses (-1).
    # - Keep log_transform_x=False; optionally z-score x for stability.
    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=False,
        z_score_theta="independent",
        z_score_x="independent",
    )

    inference = MNLE(prior=prior, density_estimator=estimator_builder, device=str(device))
    inference = inference.append_simulations(theta_train, x_train, data_device="cpu")
    density_estimator = inference.train(training_batch_size=4096)

    print("Estimator device:", next(density_estimator.parameters()).device)

    # ---- observed data (CONDITIONED) ----
    theta_true = torch.tensor([0.55, 0.2, 1.5, 2.0, 0.25], device=device, dtype=torch.float32)

    x_o = simulate_observed_session_conditioned(
        theta_true,
        num_trials=3000,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
        P=P,
        seed=123,
    )

    # ---- posterior ----
    posterior = inference.build_posterior(
        density_estimator,
        prior=prior,
        mcmc_method="slice_np_vectorized",
        mcmc_parameters=dict(num_chains=4, warmup_steps=200, thin=1, init_strategy="proposal"),
    )

    samples = posterior.sample((2000,), x=x_o.detach().cpu(), method="slice_np_vectorized")
    samples = samples.detach().cpu()

    # ---- plot ----
    labels = [r"$a_0$", r"$\lambda$", r"$v$", r"$B$", r"$t_{nd}$"]
    fig, ax = pairplot(
        [prior.sample((2000,)).detach().cpu(), samples],
        points=theta_true.detach().cpu().unsqueeze(0),
        diag="kde",
        upper="kde",
        labels=labels,
    )
    plt.show()


if __name__ == "__main__":
    main()
