"""Run inference from a pretrained NPE model (streaming-compatible architecture)."""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn

from sbi_for_diffusion_models.priors import build_prior_theta
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.Embeddings import MaskAwarePermutationInvariantEmbedding
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS
MODEL_PATH = os.path.expanduser("~/models/npe_rt_choice.pt")


def load_npe(model_path: str, device: str = "cpu"):
    """Rebuild the exact NPE architecture and load saved weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P + 1  
    theta_dim = 5
    x_dim = T * trial_dim

    embedding_net = MaskAwarePermutationInvariantEmbedding(
        num_trials=T,
        trial_dim=trial_dim,
        trial_net_hidden=int(saved_cfg.NPE_TRIAL_NET_HIDDEN),
        trial_net_layers=int(saved_cfg.NPE_TRIAL_NET_LAYERS),
        trial_net_output_dim=int(saved_cfg.NPE_TRIAL_NET_OUTPUT_DIM),
        post_agg_hidden=int(saved_cfg.NPE_POST_AGG_HIDDEN),
        post_agg_layers=int(saved_cfg.NPE_POST_AGG_LAYERS),
        output_dim=int(saved_cfg.NPE_EMBEDDING_OUTPUT_DIM),
        aggregation=str(saved_cfg.NPE_AGG_FN),
    )

    est_builder = posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=int(saved_cfg.NPE_HIDDEN_FEATURES),
        num_transforms=int(saved_cfg.NPE_NUM_TRANSFORMS),
        num_bins=int(saved_cfg.NPE_NUM_BINS),
        embedding_net=embedding_net,
    )

    # Instantiate the nn.Module with dummy data (required by sbi's builder)
    dummy_theta = torch.randn(2, theta_dim, device=device)
    dummy_x = torch.randn(2, x_dim, device=device)
    density_estimator = est_builder(dummy_theta, dummy_x)

    # Load weights
    density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    density_estimator.to(device)
    density_estimator.eval()

    return density_estimator, saved_cfg


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"Loading model from {MODEL_PATH} ...")
    density_estimator, saved_cfg = load_npe(MODEL_PATH, device=device)
    print("Model loaded.")
    print("Estimator param device:", next(density_estimator.parameters()).device)

    # Prior
    prior_theta = build_prior_theta()
    if hasattr(prior_theta, "to"):
        # IMPORTANT: do not reassign (some priors' .to() returns None)
        prior_theta.to(dev)

    # Posterior wrapper
    posterior = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior_theta,
        device=device,
    )

    # Dimensions / sim knobs must match training
    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)

    # --- simulate a "true" theta and an observed session in exactly the same format as training ---
    if bool(saved_cfg.THETA_TRUE_FROM_PRIOR):
        theta_true = prior_theta.sample((1,)).view(1, 5).to(dev, dtype=torch.float32)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    # simulate_training_sessions returns x already flattened as (N, T*(2+P+1)) with mask included
    _, x_o_flat = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=1,
        num_trials=T,
        device=dev,
        mu_sensory=float(saved_cfg.MU_SENSORY),
        p_success=float(saved_cfg.P_SUCCESS),
        P=P,
        log_rt=bool(saved_cfg.LOG_RT_MANUALLY),
        seed=123,
        theta=theta_true,  # conditional simulation
        sim_batch_size=1,  # keep it simple for one observed session
    )
    x_o_flat = x_o_flat.to(dev, dtype=torch.float32)

    print("theta_true:", theta_true.detach().cpu().view(-1).numpy().round(4).tolist())
    print("x_o_flat shape:", tuple(x_o_flat.shape), "device:", x_o_flat.device)

    # --- sample posterior ---
    n_samps = int(cfg.NPE_POSTERIOR_SAMPLES)
    print(f"\n--- Sampling posterior (direct) n={n_samps} ---")
    samples = posterior.sample(
        (n_samps,),
        x=x_o_flat,
        show_progress_bars=True,
    ).detach().cpu()

    # --- save outputs ---
    outdir = os.environ.get("OUTDIR", "npe_outputs")
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(outdir, "posterior_samples_theta.npy")
    np.save(npy_path, samples.numpy())
    print("Saved:", npy_path)

    fig, ax = pairplot(
        samples,
        points=theta_true.detach().cpu().view(1, -1),
        labels=["a0", "lam", "v", "B", "tau"],
        points_colors="r",
    )
    fig_path = os.path.join(outdir, "pairplot_theta.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()