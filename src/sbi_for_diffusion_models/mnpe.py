from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Callable 

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding

# ---------------------------------------------------------------------
# Training-data simulation
# ---------------------------------------------------------------------

@torch.no_grad()
def simulate_npe_training_data(
    cfg,
    prior_theta,
    *,
    simulate_batch_fn: Callable,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate session-level training pairs (theta, x) for NPE.
    Returns theta_train of shape (N, D) and flattened x_train of shape (N, T*(2+P)).
    """
    P = max_num_pulses()
    theta_train, x_train = simulate_training_sessions(
        prior_theta,
        num_sessions=int(cfg.NPE_NUM_SESSIONS),
        num_trials=int(cfg.NUM_TRIALS_OBS),
        simulate_batch_fn=simulate_batch_fn, 
        device=device,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(seed),
    )
    return theta_train, x_train

def _build_npe_embedding_net(cfg, *, T: int, P: int) -> torch.nn.Module:
    """Build the permutation-invariant session embedding network."""
    trial_dim = 2 + P

    return PermutationInvariantEmbedding(
        num_trials=T,
        trial_dim=trial_dim,
        trial_net_hidden=int(cfg.NPE_TRIAL_NET_HIDDEN),
        trial_net_layers=int(cfg.NPE_TRIAL_NET_LAYERS),
        trial_net_output_dim=int(cfg.NPE_TRIAL_NET_OUTPUT_DIM),
        post_agg_hidden=int(cfg.NPE_POST_AGG_HIDDEN),
        post_agg_layers=int(cfg.NPE_POST_AGG_LAYERS),
        output_dim=int(cfg.NPE_EMBEDDING_OUTPUT_DIM),
        aggregation=str(cfg.NPE_AGG_FN),
    )

def _build_npe_estimator_builder(cfg, embedding_net):
    """Build the SBI posterior density-estimator factory."""
    return posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=int(cfg.NPE_HIDDEN_FEATURES),
        num_transforms=int(cfg.NPE_NUM_TRANSFORMS),
        num_bins=int(cfg.NPE_NUM_BINS),
        embedding_net=embedding_net,
    )

def _simulate_dummy_batch(cfg, prior_theta, *, simulate_batch_fn: Callable, dev: torch.device, seed: int, T: int, P: int):
    """Simulate a small batch used to initialize estimator input shapes."""
    return simulate_training_sessions(
        prior_theta,
        num_sessions=2,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(seed),
    )

# ---------------------------------------------------------------------
# NPE training
# ---------------------------------------------------------------------

def train_npe_session(
    cfg,
    prior_theta,
    *,
    simulate_batch_fn: Callable,
    device: str = "cpu",
    seed: int = 0,
    resume_from: Optional[str] = None,
    seed_offset: int = 0,
):
    """
    Train an amortized NPE posterior on simulated RT-choice sessions.
    Simulates session batches on the fly and returns the fitted estimator and DirectPosterior.
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)

    embedding_net = _build_npe_embedding_net(cfg, T=T, P=P).to(dev)
    est_builder = _build_npe_estimator_builder(cfg, embedding_net)

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    theta_dummy, x_dummy = _simulate_dummy_batch(
        cfg,
        prior_theta,
        simulate_batch_fn=simulate_batch_fn,
        dev=dev,
        seed=int(seed),
        T=T,
        P=P,
    )
    theta_dummy = theta_dummy.to(dev, dtype=torch.float32)
    x_dummy = x_dummy.to(dev, dtype=torch.float32)

    density_estimator = est_builder(theta_dummy, x_dummy).to(dev)

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[NPE] Resumed weights from {resume_from}")

    density_estimator.train()

    lr = float(getattr(cfg, "NPE_LR", 5e-4))
    optimizer = torch.optim.Adam(density_estimator.parameters(), lr=lr)

    sess_per_step = int(getattr(cfg, "NPE_SESSIONS_PER_STEP", 8))
    num_steps = int(getattr(cfg, "NPE_NUM_STEPS", 10_000))

    print(f"[NPE] device={dev}, sess_per_step={sess_per_step}, num_steps={num_steps}, lr={lr}")

    best = float("inf")
    bad = 0
    patience = int(getattr(cfg, "NPE_PATIENCE", 30))
    min_delta = float(getattr(cfg, "NPE_MIN_DELTA", 1e-3))
    ema_beta = float(getattr(cfg, "NPE_EMA_BETA", 0.98))
    ema = None

    for step in range(num_steps):
        theta_b, x_b = simulate_training_sessions(
            prior_theta,
            num_sessions=sess_per_step,
            num_trials=T,
            simulate_batch_fn=simulate_batch_fn,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
            P=P,
            log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=int(seed + step + seed_offset + 1),
        )
        theta_b = theta_b.to(dev, dtype=torch.float32)
        x_b = x_b.to(dev, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        losses = density_estimator.loss(theta_b, condition=x_b)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

        li = float(loss.item())
        ema = li if ema is None else ema_beta * ema + (1.0 - ema_beta) * li

        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: loss={li:.4f} ema={ema:.4f}")

            if ema < best - min_delta:
                best = ema
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"[NPE] early stop at step {step + 1}")
                    break
    
    density_estimator.eval()
    posterior = DirectPosterior(density_estimator, prior_theta)
    return density_estimator, posterior


# ---------------------------------------------------------------------
# Posterior inference
# ---------------------------------------------------------------------

def run_inference_npe(cfg, inference_obj, density_estimator, x_o_flat, prior_theta):
    """
    Draw posterior samples from the amortized NPE model for one observed session.
    Expects x_o_flat to match the flattened training representation used during fitting.
    """
    posterior = inference_obj.build_posterior(
        density_estimator=density_estimator,
        prior=prior_theta,
    )

    dev = next(density_estimator.parameters()).device
    x_o_flat = x_o_flat.to(dev, dtype=torch.float32)

    samples = (
        posterior.sample(
            (int(cfg.NPE_POSTERIOR_SAMPLES),),
            x=x_o_flat,
            show_progress_bars=True,
        )
        .detach()
        .cpu()
    )
    return samples

# ---------------------------------------------------------------------
# Load pretrained
# ---------------------------------------------------------------------
def load_npe(
    model_path: str,
    *,
    prior_theta,
    device: str = "cpu",
):
    """Rebuild the exact NPE architecture and load saved weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    dev = torch.device(device)

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P
    x_dim = T * trial_dim

    theta_probe = torch.as_tensor(
        prior_theta.sample((1,)),
        device=dev,
        dtype=torch.float32,
    ).reshape(-1)
    theta_dim = int(theta_probe.numel())

    embedding_net = PermutationInvariantEmbedding(
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

    # SBI builder requires dummy inputs to instantiate the nn.Module.
    dummy_theta = torch.randn(2, theta_dim, device=dev)
    dummy_x = torch.randn(2, x_dim, device=dev)
    density_estimator = est_builder(dummy_theta, dummy_x)

    density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    density_estimator.to(dev)
    density_estimator.eval()

    return density_estimator, saved_cfg


# ---------------------------------------------------------------------
# SBC utilities
# ---------------------------------------------------------------------

def _compute_ranks(theta_true: torch.Tensor, posterior_samples: torch.Tensor) -> torch.Tensor:
    """Compute SBC ranks for each parameter dimension."""
    theta_true = theta_true.reshape(-1)
    return (posterior_samples < theta_true[None, :]).sum(dim=0).to(torch.int64)


def _plot_sbc_rank_histograms(
    ranks: np.ndarray,
    *,
    param_names: Optional[Sequence[str]] = None, 
    outpath: Optional[str] = None,
    bins: int = 30,
):
    """Plot per-parameter SBC rank histograms."""
    D = ranks.shape[1]
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.5 * D), constrained_layout=True)
    if D == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.hist(ranks[:, d], bins=bins)
        ax.set_title(f"SBC ranks: {param_names[d]}")
        ax.set_xlabel("rank")
        ax.set_ylabel("count")

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print("Saved SBC plot:", outpath)

    return fig


@torch.no_grad()
def run_sbc_npe(
    cfg,
    *,
    prior_theta,
    posterior,
    simulate_batch_fn: Callable,
    device: str = "cpu",
    num_datasets: int = 100,
    posterior_samples_per_dataset: Optional[int] = None,
    seed: int = 0,
    param_names: Optional[Sequence[str]] = None, 
    outdir: str = "sbc_npe_outputs",
    plot_bins: int = 30,
) -> dict:
    """
    Run simulation-based calibration for the NPE pipeline.
    Simulates datasets from the prior, samples posteriors, and saves rank diagnostics.
    """
    os.makedirs(outdir, exist_ok=True)

    dev = torch.device(device)
    S = int(posterior_samples_per_dataset or cfg.NPE_SBC_POST_SAMPLES)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    thetas_true = []
    ranks = []
    all_samples = []

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
    trial_dim = 2 + P

    # infer theta dimension from prior
    theta_probe = torch.as_tensor(
        prior_theta.sample((1,)),
        device=dev,
        dtype=torch.float32,
    ).reshape(-1)
    D = int(theta_probe.numel())

    if param_names is None:
        param_names = tuple(f"theta_{d}" for d in range(D))
    else:
        param_names = tuple(param_names)
        if len(param_names) != D:
            raise ValueError(f"len(param_names)={len(param_names)} but theta_dim={D}")

    for i in range(int(num_datasets)):
        theta_true = torch.as_tensor(
            prior_theta.sample((1,)),
            device=dev,
            dtype=torch.float32,
        ).reshape(-1)

        _, x_o = simulate_training_sessions(
            prior_theta,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=simulate_batch_fn,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
            P=P,
            log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=int(rng.integers(0, 2**31 - 1)),
            theta=theta_true,
        )

        if x_o.shape != (1, T * trial_dim):
            raise RuntimeError(f"Unexpected x_o shape: {tuple(x_o.shape)}")

        samples = posterior.sample((S,), x=x_o, show_progress_bars=False).detach().cpu()
        r = _compute_ranks(theta_true.detach().cpu(), samples)

        thetas_true.append(theta_true.detach().cpu().numpy())
        ranks.append(r.numpy())
        all_samples.append(samples)

        if (i + 1) % 10 == 0:
            print(f"[SBC] {i + 1}/{num_datasets} done. ranks={r.tolist()}")

    thetas_true = np.stack(thetas_true, axis=0)
    ranks = np.stack(ranks, axis=0)

    np.save(os.path.join(outdir, "sbc_thetas_true.npy"), thetas_true)
    np.save(os.path.join(outdir, "sbc_ranks.npy"), ranks)
    print("Saved:", os.path.join(outdir, "sbc_thetas_true.npy"))
    print("Saved:", os.path.join(outdir, "sbc_ranks.npy"))

    _plot_sbc_rank_histograms(
        ranks,
        param_names=param_names,
        outpath=os.path.join(outdir, "sbc_rank_histograms.png"),
        bins=int(plot_bins),
    )

    return {"thetas_true": thetas_true, "ranks": ranks, "all_samples": all_samples}