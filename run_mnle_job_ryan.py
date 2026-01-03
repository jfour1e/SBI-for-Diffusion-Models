import torch
import matplotlib.pyplot as plt

from sbi.inference import MNLE
from sbi.analysis import pairplot
from sbi.utils.get_nn_models import likelihood_nn

from torch.distributions import Normal, Beta, LogNormal, TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch.distributions import Distribution

from sbi_for_diffusion_models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    simulate_session_data_rt_choice,
)

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

        # Hard support checks
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

@torch.no_grad()
def simulate_training_set(prior, num_simulations: int, batch_size: int, device, **sim_kwargs):
    theta = prior.sample((num_simulations,)).to(dtype=torch.float32)  # on device

    xs = []
    for start in range(0, num_simulations, batch_size):
        th = theta[start:start + batch_size]
        x = rt_choice_model_simulator_torch(th, **sim_kwargs)
        xs.append(x.detach().cpu())

    x = torch.cat(xs, dim=0).to(torch.float32)
    theta_cpu = theta.detach().cpu()

    assert torch.isfinite(theta_cpu).all()
    assert torch.isfinite(x).all()
    assert torch.all((x[:, 1] == 0) | (x[:, 1] == 1) | (x[:, 1] == 2))
    print("Unique outcomes in training x:", x[:, 1].unique().tolist())
    return theta_cpu, x


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

    # ---- simulate ----
    theta_train, x_train = simulate_training_set(
        prior,
        num_simulations=50_000,
        batch_size=4096,
        device=device,
        mu_sensory=1.0,
        p_success=0.75,
    )

    # ---- build & train MNLE ----
    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=True,
        z_score_theta="independent",
        z_score_x="independent",
    )

    inference = MNLE(prior=prior, density_estimator=estimator_builder, device=str(device))
    inference = inference.append_simulations(theta_train, x_train, data_device="cpu")
    density_estimator = inference.train(training_batch_size=4096)

    print("Estimator device:", next(density_estimator.parameters()).device)

    # ---- observed data ----
    theta_true = torch.tensor([0.55, 0.2, 1.5, 2.0, 0.25], device=device, dtype=torch.float32)
    x_o = simulate_session_data_rt_choice(theta_true, num_trials=300, mu_sensory=1.0, p_success=0.75)
    print("Observed unique outcomes:", x_o[:, 1].unique().tolist())

    # ---- IMPORTANT: do not call 'nuts' here; use slice or nuts_pyro if installed ----
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
    # This line matters on Windows when subprocesses are spawned.
    main()
