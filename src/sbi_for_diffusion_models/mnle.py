from __future__ import annotations
import torch
from torch import Tensor
import sys, os

import sbi.neural_nets as nn
from sbi.inference import MNLE
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from sbi.utils.get_nn_models import likelihood_nn

from sbi_for_diffusion_models.ddm_simulator import pulse_ddm_simulator_torch, simulate_session_data

def train_mnle(
    prior,
    num_simulations: int = 10_000,
    simulation_batch_size: int = 512,
    mcmc_kwargs: dict | None = None,
    **sim_kwargs, 
):
    if mcmc_kwargs is None:
        mcmc_kwargs = dict(
            num_chains=20,
            warmup_steps=200,
            thin=5,
            init_strategy="proposal",
        )
    
    theta_train = prior.sample((num_simulations,)).to(torch.float32)

    x_train_list = []
    for start in range(0, num_simulations, simulation_batch_size):
        batch = theta_train[start:start + simulation_batch_size]
        x_batch = pulse_ddm_simulator_torch(batch, **sim_kwargs) 
        x_train_list.append(x_batch)

    x_train = torch.cat(x_train_list, dim=0).to(torch.float32)

    assert torch.isfinite(theta_train).all(), "NaN/Inf in theta_train."
    assert torch.isfinite(x_train).all(), "NaN/Inf in x_train."

    estimator_builder = likelihood_nn(
        model="mnle",
        log_transform_x=True,
        z_score_theta="independent",
        z_score_x="independent",
    )

    trainer = MNLE(prior=prior, density_estimator=estimator_builder)
    trainer.append_simulations(theta_train, x_train, exclude_invalid_x=False)
    trainer.train()

    posterior = trainer.build_posterior(
        prior=prior,
        mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_kwargs,
    )

    return posterior, trainer


def run_parameter_recovery(
    prior,
    num_simulations=50_000,
    num_trials_session=200,
    num_posterior_samples=5_000,
    theta_true=None,
    mu_sensory: float = 1.0,
):
    """
    Train once, simulate one synthetic dataset, sample posterior, plot pairplot.
    """
    mcmc_kwargs = dict(
        method="slice_np_vectorized",
        warmup_steps=200,
        thin=5,
        num_chains=20,
        init_strategy="proposal",
    )

    posterior, trainer = train_mnle(
        prior,
        num_simulations=num_simulations,
        mu_sensory=mu_sensory,    
    )

    if theta_true is None:
        theta_true = torch.tensor(
            [0.5, 0.4, 1.0, 2.0, 0.2, 0.2, 0.15],  # [bias, lam, nu, B, sigma_a, t_nd, sigma_s]
            dtype=torch.float32
        )

    x_o = simulate_session_data(theta_true, num_trials_session, mu_sensory=mu_sensory)

    # ---- Sampling from MCMCPosterior: pass kwargs here (works across versions)
    posterior_samples = posterior.sample(
        sample_shape=(num_posterior_samples,),
        x=x_o,
        **mcmc_kwargs,
    )

    labels = [r"bias", r"$\lambda$", r"$\nu$", r"$B$", r"$\sigma_a$", r"$t_{nd}$", r"$\sigma_s$"]

    fig, ax = pairplot(
        [prior.sample((2000,)), posterior_samples],
        points=theta_true.unsqueeze(0),
        diag="kde",
        upper="kde",
        labels=labels,
    )
    plt.suptitle("Parameter Recovery: MNLE posterior vs prior", fontsize=14)
    plt.show()

    return theta_true, x_o, posterior, posterior_samples, trainer


def plot_empirical_rt_choice(x_o: torch.Tensor, title: str = "Observed data"):
    x_np = x_o.detach().cpu().numpy()
    rt = x_np[:, 0]
    choice = x_np[:, 1]
    p_right = choice.mean()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(rt, bins=30, alpha=0.8)
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title}: RT distribution")

    axes[1].bar(["Left", "Right"], [1.0 - p_right, p_right])
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{title}: choice proportions")

    plt.tight_layout()
    plt.show()


def plot_posterior_marginals(posterior_samples: torch.Tensor):
    labels = [r"bias", r"$\lambda$", r"$\nu$", r"$B$", r"$\sigma_a$", r"$t_{nd}$", r"$\sigma_s$"]
    s = posterior_samples.detach().cpu().numpy()

    fig, axes = plt.subplots(1, s.shape[1], figsize=(3.0 * s.shape[1], 3))
    for i in range(s.shape[1]):
        axes[i].hist(s[:, i], bins=30, alpha=0.8)
        axes[i].set_title(labels[i])
    plt.tight_layout()
    plt.show()
