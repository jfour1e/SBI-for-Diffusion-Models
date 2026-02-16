
from __future__ import annotations

"""
Run the rt_choice_model pipeline starting from a *saved* (pretrained) MNLE network.

This mirrors `rt_choice_model_pipeline.py` but skips simulation+training and instead:
  1) rebuilds the MNLE architecture
  2) loads weights from a checkpoint
  3) simulates (x_o, pulses_o)
  4) runs MCMC over global theta (dim=5)
  5) saves posterior samples + pairplot

Checkpoint formats supported:
  A) {"density_estimator": <torch.nn.Module>, "config": cfg, ...}  (preferred)
  B) {"state_dict": <state_dict>, "config": cfg, ...}             (supported)

For format (B), we reconstruct the MNLE network by creating the same likelihood_nn builder
and *materializing* the network with a tiny dummy append_simulations call before loading weights.
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference import MNLE
from sbi.neural_nets import likelihood_nn
from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.proposals import PulseSequenceProposal, ExtendedProposal
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.mnle import run_inference_mcmc
from sbi_for_diffusion_models.data_simulator import simulate_observed_session, summarize_trials
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_checkpoint_path(model_name: str | None = None) -> Path:
    """
    If model_name is provided, load that file from models/.
    Otherwise load the most recent .pt file.
    """
    if model_name is not None:
        ckpt_path = MODELS_DIR / model_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} not found.")
        return ckpt_path

    # otherwise load most recent
    pt_files = sorted(MODELS_DIR.glob("*.pt"), key=os.path.getmtime)
    if not pt_files:
        raise RuntimeError("No .pt files found in models/")
    return pt_files[-1]


def load_checkpoint(path: Path):
    print(f"\n--- Loading pretrained MNLE from {path.name} ---")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _build_mnle_estimator(cfg, proposal_z, *, device: str = "cpu", P: int) -> torch.nn.Module:
    """
    Reconstruct an *untrained* MNLE density estimator with the same architecture as training.

    Key detail: the MNLE / mixed density estimator builder may infer transforms (z-scoring,
    log-transform validity masks, number of categories) from the batch you pass in.

    So we "materialize" the network with a tiny **valid** dummy batch:
      - RT must be > 0 if log transforms are used.
      - Include at least two choices (0 and 1) so num_categories is inferred correctly.
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

    # Dummy batch (N=4) to make statistics well-defined.
    # z is (N, 5+P), x is (N, 2) = [rt, choice]
    dummy_z = torch.zeros((4, 5 + int(P)), dtype=torch.float32, device=device)

    # Use strictly positive RTs to avoid invalid log() / masking.
    dummy_rt = torch.tensor([[0.6], [0.9], [1.2], [1.5]], dtype=torch.float32, device=device)
    dummy_choice = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32, device=device)
    dummy_x = torch.cat([dummy_rt, dummy_choice], dim=1)

    # Many sbi versions: likelihood_nn(...) returns a callable that builds the net from (theta, x).
    # We try both common calling conventions.
    density_estimator = None
    for call in (
        lambda: est_builder(dummy_z, dummy_x),
        lambda: est_builder(theta=dummy_z, x=dummy_x),
    ):
        try:
            density_estimator = call()
            break
        except TypeError:
            continue

    # Fallback: materialize via MNLE.append_simulations (older sbi versions).
    if density_estimator is None:
        trainer = MNLE(prior=proposal_z, density_estimator=est_builder, device=device)
        trainer = trainer.append_simulations(dummy_z, dummy_x)
        density_estimator = getattr(trainer, "_neural_net", None) or getattr(trainer, "neural_net", None)

    if density_estimator is None:
        raise RuntimeError(
            "Could not materialize MNLE density estimator. "
            "Your sbi version likely changed its builder API. "
            "Try saving checkpoints with ckpt['density_estimator']=inference._neural_net."
        )

    density_estimator.to(device)
    density_estimator.eval()
    return density_estimator


def load_pretrained_density_estimator(
    ckpt,
    *,
    proposal_z,
    cfg,
    device: str,
    P: int,
) -> torch.nn.Module:
    """
    ckpt: already-loaded checkpoint object (dict or nn.Module).
    proposal_z/cfg/P only used if ckpt contains a state_dict and we must rebuild the net.
    """

    # Preferred: checkpoint contains the whole module
    if isinstance(ckpt, dict) and "density_estimator" in ckpt:
        density_estimator = ckpt["density_estimator"]
        density_estimator.to(device)
        density_estimator.eval()
        return density_estimator

    # Common: checkpoint stores a state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        density_estimator = _build_mnle_estimator(cfg, proposal_z, device=device, P=P)
        missing, unexpected = density_estimator.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print("[warn] load_state_dict reported:")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)
        density_estimator.eval()
        return density_estimator

    # Some users save the module directly via torch.save(module, path)
    if hasattr(ckpt, "state_dict") and callable(getattr(ckpt, "state_dict")):
        density_estimator = ckpt
        density_estimator.to(device)
        density_estimator.eval()
        return density_estimator

    raise ValueError(
        "Unrecognized checkpoint format. Expected dict with 'density_estimator' or 'state_dict', "
        "or a torch module."
    )


def main():
    ckpt_path = get_checkpoint_path()

    ckpt = load_checkpoint(ckpt_path)

    torch.manual_seed(0)
    np.random.seed(0)

    # Determine pulse length P from time discretization (must match training)
    P = max_num_pulses()
    print("P =", P, "pulses per trial")

    # Prior over theta
    prior_theta = build_prior_theta()

    # Proposal over z=[theta,pulses] (needed only because MNLE expects a prior/proposal over conditions)
    pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0, device="cpu")
    proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

    # Load pretrained MNLE
    print("\n--- Loading pretrained MNLE ---")
    density_estimator = load_pretrained_density_estimator(
        ckpt,
        proposal_z=proposal_z,
        cfg=cfg,
        P=P,
        device=DEVICE,
    )

    # Simulate observed session data (same as pipeline)
    if cfg.THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=cfg.NUM_TRIALS_OBS,
        device=DEVICE,
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # Inference via MCMC (theta-only)
    print("\n--- Sampling posterior over theta (starting from pretrained MNLE) ---")
    samples = run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o)

    outdir = MODELS_DIR / f"inference_{ckpt_path.stem}"

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
