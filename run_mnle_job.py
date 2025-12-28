from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from pathlib import Path

# set up directory for calling from src
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from sbi_for_diffusion_models.ddm_simulator import pulse_ddm_simulator_torch, simulate_session_data
from sbi_for_diffusion_models.mnle import train_mnle, run_parameter_recovery
from sbi.utils import MultipleIndependent
from torch.distributions import Beta, Uniform, LogNormal, HalfNormal

# define prior for each argument in theta 
def make_prior():
    prior = MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),              # bias
            HalfNormal(torch.tensor([0.5])),                             # lam
            LogNormal(torch.tensor([np.log(1.0)]), torch.tensor([0.5])),  # nu
            LogNormal(torch.tensor([np.log(2.0)]), torch.tensor([0.5])),  # B
            HalfNormal(torch.tensor([0.5])),                             # sigma_a
            Uniform(torch.tensor([0.05]), torch.tensor([0.6])),          # t_nd
            HalfNormal(torch.tensor([0.5])),                             # sigma_s
        ],
        validate_args=False,
    )
    return prior

def main():
    prior = make_prior()

    # define trials, sessions and posterior samples for qsub job 
    p = argparse.ArgumentParser()
    p.add_argument("--num_simulations", type=int, default=1_000)
    p.add_argument("--num_trials_session", type=int, default=10)
    p.add_argument("--num_posterior_samples", type=int, default=500)
    p.add_argument("--outdir", type=str, default="outputs/latest")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # run parameter recovery experiment  
    theta_true, x_o, posterior, posterior_samples, trainer = run_parameter_recovery(
        prior = prior,
        num_simulations=args.num_simulations,
        num_trials_session=args.num_trials_session,
        num_posterior_samples=args.num_posterior_samples,
        theta_true=None,
        mu_sensory=1.0,
    )
 
    torch.save(
        {
            "theta_true": theta_true,
            "x_o": x_o,
            "posterior_samples": posterior_samples,
        },
        os.path.join(args.outdir, "recovery_outputs.pt"),
    )

    estimator = getattr(trainer, "_neural_net", None) or getattr(trainer, "_density_estimator", None)
    if estimator is None:
        raise RuntimeError("Could not find trained density estimator on trainer.")

    # freeze weights 
    estimator.eval()
    for p in estimator.parameters():
        p.requires_grad_(False)
    
    # save torch weights (of Likelihood NN) to directory
    torch.save(estimator.state_dict(), os.path.join(args.outdir, "mnle_estimator_state_dict.pt"))
    
    # simple marginals plot
    labels = ["bias","lam","nu","B","sigma_a","t_nd","sigma_s"]
    s = posterior_samples.detach().cpu().numpy()
    fig, axes = plt.subplots(1, s.shape[1], figsize=(3.0 * s.shape[1], 3))
    for i in range(s.shape[1]):
        axes[i].hist(s[:, i], bins=40, alpha=0.8)
        axes[i].set_title(labels[i])
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "posterior_param0.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()