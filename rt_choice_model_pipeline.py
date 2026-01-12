from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Avoid expensive distribution argument checks in tight loops.
torch.distributions.Distribution.set_default_validate_args(False)

from torch.distributions import Beta, LogNormal, Distribution
from sbi.analysis import pairplot
from sbi.utils import MultipleIndependent

from sbi_for_diffusion_models.proposals import PulseSequenceProposal, ExtendedProposal
from sbi_for_diffusion_models.models.rt_choice_model import (
    pulse_schedule,
    n_pulses_max_from_schedule,
)
from sbi_for_diffusion_models.mnle import train_mnle, run_inference_mcmc
from sbi_for_diffusion_models.data_simulator import (
    simulate_observed_session, 
    simulate_training_set_with_conditions, 
    summarize_trials
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, RunConfig
cfg = RUN_CONFIG_PARAMS

# -------------
# Utilities 
# -------------
def build_prior_theta(device: str = "cpu") -> Distribution:
    """
    Prior over theta = [a0, lam, v, B, tau].
    """
    return MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),          # a0
            LogNormal(torch.tensor([-1.0]), torch.tensor([1.0])),     # lam
            LogNormal(torch.tensor([0.0]), torch.tensor([1.0])),      # v
            LogNormal(torch.tensor([2.75]), torch.tensor([0.5])),     # B
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),           # tau (placeholder Beta)
        ]
    )

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # Determine pulse length P from time discretization
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    # prior over Theta 
    prior_theta = build_prior_theta(device="cpu")

    # Training proposal over z=[theta,pulses]
    pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0, device="cpu")
    proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

    # simualte training set
    print("\n--- Simulating training set ---")
    z_train, x_train = simulate_training_set_with_conditions(
        proposal=proposal_z,
        num_simulations=cfg.NUM_SIMULATIONS,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        device="cpu",
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        log_rt=cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("train (sample)", x_train[torch.randperm(len(x_train))[:50_000]])

    # Train MNLE
    print("\n--- Training MNLE ---")
    density_estimator = train_mnle(cfg, proposal_z, z_train, x_train, device="cpu")

    # Simulate observed session data
    if cfg.THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")
    
    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=cfg.NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # Inference 
    print("\n--- Sampling posterior over theta ---")
    samples = run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o)

    # Save outputs
    outdir = os.environ.get("OUTDIR", "mnle_outputs")
    os.makedirs(outdir, exist_ok=True)
   
    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

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

if __name__ == "__main__":
    main()