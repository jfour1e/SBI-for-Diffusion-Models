from __future__ import annotations

import torch
from sbi.inference import MNLE, MCMCPosterior
from sbi.neural_nets import likelihood_nn
from sbi.utils import mcmc_transform

from sbi_for_diffusion_models.potentials import ThetaOnlyPosteriorPotential, ConditionedMNLELogLikelihood

def train_mnle(cfg, proposal_z, z_train, x_train, device: str = "cpu"):
    """
    Train an MNLE density estimator on simulations (z_train, x_train).

    Args:
        cfg: RunConfig (or any object with the required attributes)
        proposal_z: distribution over z = [theta, pulses]
        z_train: (N, 5+P) tensor
        x_train: (N, 2) tensor
        device: "cpu" or "cuda"

    Returns:
        density_estimator: trained conditional density estimator
    """

    est_builder = likelihood_nn(
        model="mnle",
        log_transform_x=bool(cfg.SBI_LOG_TRANSFORM_X),
        z_score_theta="independent",
        z_score_x=cfg.Z_SCORE_X,
        hidden_features=128,
        num_transforms=10,
        num_bins=24,
    )

    trainer = MNLE(prior=proposal_z, density_estimator=est_builder, device=device)
    trainer = trainer.append_simulations(z_train, x_train)

    # sbi changed arg name across versions
    try:
        density_estimator = trainer.train(training_batch_size=cfg.TRAIN_BATCH_SIZE)
    except TypeError:
        density_estimator = trainer.train(batch_size=cfg.TRAIN_BATCH_SIZE)

    return density_estimator

def run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o) -> torch.Tensor:
    """
    Runs MCMC over global theta (dim=5) conditioned on trial-wise pulses_o,
    using an MNLE density_estimator trained on z=[theta,pulses].

    Returns:
        samples: (cfg.POSTERIOR_SAMPLES, 5) on CPU
    """

    conditioned_loglike = ConditionedMNLELogLikelihood(
        estimator=density_estimator,
        local_theta=pulses_o,
        device="cpu",
    )

    potential_theta = ThetaOnlyPosteriorPotential(
        conditioned_loglike=conditioned_loglike,
        prior_theta=prior_theta,
        x_o=x_o,
        device="cpu",
        temperature=cfg.TEMPERATURE,
    )

    theta_transform = mcmc_transform(prior_theta)

    posterior = MCMCPosterior(
        potential_fn=potential_theta,
        proposal=prior_theta,
        theta_transform=theta_transform,
        method="nuts_pyro",
        num_chains=cfg.NUM_CHAINS,
        warmup_steps=cfg.WARMUP_STEPS,
        thin=1,
        init_strategy="proposal",
        num_workers=1,
    )

    samples = posterior.sample(
        (cfg.POSTERIOR_SAMPLES,),
        x=x_o,
        show_progress_bars=True,
    ).detach().cpu()

    return samples