from __future__ import annotations

import os
from typing import Optional, Sequence, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.neural_nets import posterior_nn
from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions,
    simulate_training_sessions_ar,
    trial_feature_dim,
    load_corpus,
)
from sbi_for_diffusion_models.Embeddings import PermutationInvariantEmbedding


def _pick_session_simulator(autoregressive: bool) -> Callable:
    return simulate_training_sessions_ar if autoregressive else simulate_training_sessions


def _p_success_training_values(cfg):
    return tuple(getattr(cfg, "P_SUCCESS_TRAIN_VALUES", (float(cfg.P_SUCCESS),)))


def _build_npe_embedding_net(cfg, *, T: int, P: int, autoregressive: bool = False) -> torch.nn.Module:
    trial_dim = trial_feature_dim(P, ar=autoregressive)

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
    return posterior_nn(
        model="nsf",
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=int(cfg.NPE_HIDDEN_FEATURES),
        num_transforms=int(cfg.NPE_NUM_TRANSFORMS),
        num_bins=int(cfg.NPE_NUM_BINS),
        embedding_net=embedding_net,
    )


def _simulate_dummy_batch(cfg, prior_theta, *, simulate_batch_fn: Callable, dev: torch.device, seed: int, T: int, P: int, autoregressive: bool = False):
    session_fn = _pick_session_simulator(autoregressive)
    return session_fn(
        prior_theta,
        num_sessions=2,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=_p_success_training_values(cfg),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(seed),
    )


def train_npe_session(
    cfg,
    prior_theta,
    *,
    simulate_batch_fn: Callable,
    device: str = "cpu",
    seed: int = 0,
    resume_from: Optional[str] = None,
    seed_offset: int = 0,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 0,
    autoregressive: bool = False,
):
    dev = torch.device(device)
    torch.manual_seed(int(seed))

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)

    embedding_net = _build_npe_embedding_net(cfg, T=T, P=P, autoregressive=autoregressive).to(dev)
    est_builder = _build_npe_estimator_builder(cfg, embedding_net)
    session_fn = _pick_session_simulator(autoregressive)

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
        autoregressive=autoregressive,
    )
    theta_dummy = theta_dummy.to(dev, dtype=torch.float32)
    x_dummy = x_dummy.to(dev, dtype=torch.float32)

    density_estimator = est_builder(theta_dummy, x_dummy).to(dev)

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[NPE] Resumed weights from {resume_from}")

    val_sessions = int(getattr(cfg, "NPE_VAL_SESSIONS", 0))
    val_every = int(getattr(cfg, "NPE_VAL_EVERY", 100))
    val_patience = int(getattr(cfg, "NPE_VAL_PATIENCE", 20))
    val_min_delta = float(getattr(cfg, "NPE_VAL_MIN_DELTA", 1e-3))
    val_batch = int(getattr(cfg, "NPE_VAL_BATCH", 512))
    val_seed_offset = int(getattr(cfg, "NPE_VAL_SEED_OFFSET", 999_983))
    use_val = val_sessions > 0 and val_every > 0

    theta_val = x_val = None
    if use_val:
        print(f"[NPE] Building held-out val set: {val_sessions} sessions ...")
        theta_val, x_val = session_fn(
            prior_theta,
            num_sessions=val_sessions,
            num_trials=T,
            simulate_batch_fn=simulate_batch_fn,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=_p_success_training_values(cfg),
            P=P,
            log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=int(seed + val_seed_offset),
            warn_on_timeouts=False,
        )
        theta_val = theta_val.to(dev, dtype=torch.float32)
        x_val = x_val.to(dev, dtype=torch.float32)

    density_estimator.train()

    lr = float(getattr(cfg, "NPE_LR", 5e-4))
    optimizer = torch.optim.AdamW(density_estimator.parameters(), lr=lr)

    sess_per_step = int(getattr(cfg, "NPE_SESSIONS_PER_STEP", 8))
    num_steps = int(getattr(cfg, "NPE_NUM_STEPS", 10_000))

    print(
        f"[NPE] device={dev}, sess_per_step={sess_per_step}, "
        f"num_steps={num_steps}, lr={lr}, val={'on' if use_val else 'off'}"
    )

    ema_beta = float(getattr(cfg, "NPE_EMA_BETA", 0.98))
    ema = None

    best_val = float("inf")
    bad_val = 0
    best_train = float("inf")
    bad_train = 0
    train_patience = int(getattr(cfg, "NPE_PATIENCE", 30))
    train_min_delta = float(getattr(cfg, "NPE_MIN_DELTA", 1e-3))
    best_state_dict = None

    sim_seed_base = int(seed + seed_offset + 1)

    def _compute_val_loss() -> float:
        density_estimator.eval()
        with torch.no_grad():
            n = theta_val.shape[0]
            total = 0.0
            count = 0
            for s in range(0, n, val_batch):
                e = min(n, s + val_batch)
                losses = density_estimator.loss(
                    theta_val[s:e], condition=x_val[s:e]
                )
                total += float(losses.sum().item())
                count += int(e - s)
        density_estimator.train()
        return total / max(1, count)

    for step in range(num_steps):

        with torch.no_grad():
            theta_b, x_b = session_fn(
                prior_theta,
                num_sessions=sess_per_step,
                num_trials=T,
                simulate_batch_fn=simulate_batch_fn,
                device=dev,
                mu_sensory=float(cfg.MU_SENSORY),
                p_success=_p_success_training_values(cfg),
                P=P,
                log_rt=bool(cfg.LOG_RT_MANUALLY),
                seed=sim_seed_base + step,
                warn_on_timeouts=False,
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

        if use_val and (step + 1) % val_every == 0:
            val_loss = _compute_val_loss()
            improved = val_loss < best_val - val_min_delta
            marker = "*" if improved else " "
            print(
                f"[VAL] step {step + 1}: val_loss={val_loss:.4f} "
                f"(best={best_val:.4f}) {marker}"
            )
            if improved:
                best_val = val_loss
                bad_val = 0
                best_state_dict = {
                    k: v.detach().clone() for k, v in density_estimator.state_dict().items()
                }
            else:
                bad_val += 1
                if bad_val >= val_patience:
                    print(
                        f"[NPE] early stop at step {step + 1} "
                        f"(no val improvement for {val_patience} evals)"
                    )
                    break
        elif not use_val and (step + 1) % 50 == 0:
            if ema < best_train - train_min_delta:
                best_train = ema
                bad_train = 0
            else:
                bad_train += 1
                if bad_train >= train_patience:
                    print(f"[NPE] early stop (EMA) at step {step + 1}")
                    break

        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and (step + 1) % int(checkpoint_every) == 0
        ):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": density_estimator.state_dict(),
                    "config": cfg,
                    "step": step + 1,
                    "best_val": best_val if use_val else None,
                },
                checkpoint_path,
            )
            print(f"[NPE] checkpoint saved at step {step + 1}: {checkpoint_path}")

    if use_val and best_state_dict is not None:
        density_estimator.load_state_dict(best_state_dict, strict=True)
        print(f"[NPE] Restored best-val weights (val_loss={best_val:.4f})")

    density_estimator.eval()
    posterior = DirectPosterior(density_estimator, prior_theta)
    return density_estimator, posterior


def train_npe_session_from_corpus(
    cfg,
    prior_theta,
    *,
    corpus_train_dir: str,
    corpus_val_dir: Optional[str] = None,
    device: str = "cpu",
    seed: int = 0,
    resume_from: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 0,
    autoregressive: bool = False,
):
    """Train NPE from a pre-simulated corpus on disk.

    `corpus_train_dir` and (optional) `corpus_val_dir` are directories of chunk
    `.pt` files written by `pre_simulate.py`. All chunks must share theta_dim,
    x_dim, and autoregressive flag.
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)

    embedding_net = _build_npe_embedding_net(cfg, T=T, P=P, autoregressive=autoregressive).to(dev)
    est_builder = _build_npe_estimator_builder(cfg, embedding_net)

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    print(f"[NPE] loading train corpus from {corpus_train_dir} ...")
    theta_train, x_train = load_corpus(corpus_train_dir, autoregressive=autoregressive, pattern="chunk_*.pt")
    theta_train = theta_train.to(dev, dtype=torch.float32)
    x_train = x_train.to(dev, dtype=torch.float32)

    N_train = theta_train.shape[0]
    print(f"[NPE] train corpus N={N_train}, theta_dim={theta_train.shape[1]}, x_dim={x_train.shape[1]}")

    theta_val = x_val = None
    use_val = False
    if corpus_val_dir is not None and os.path.isdir(corpus_val_dir):
        print(f"[NPE] loading val corpus from {corpus_val_dir} ...")
        theta_val, x_val = load_corpus(corpus_val_dir, autoregressive=autoregressive, pattern="val_*.pt")
        theta_val = theta_val.to(dev, dtype=torch.float32)
        x_val = x_val.to(dev, dtype=torch.float32)
        use_val = True

    density_estimator = est_builder(theta_train[:2], x_train[:2]).to(dev)

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
        print(f"[NPE] Resumed weights from {resume_from}")

    val_every = int(getattr(cfg, "NPE_VAL_EVERY", 100))
    val_patience = int(getattr(cfg, "NPE_VAL_PATIENCE", 20))
    val_min_delta = float(getattr(cfg, "NPE_VAL_MIN_DELTA", 1e-3))
    val_batch = int(getattr(cfg, "NPE_VAL_BATCH", 512))

    density_estimator.train()

    lr = float(getattr(cfg, "NPE_LR", 5e-4))
    optimizer = torch.optim.AdamW(density_estimator.parameters(), lr=lr)

    sess_per_step = int(getattr(cfg, "NPE_SESSIONS_PER_STEP", 8))
    num_steps = int(getattr(cfg, "NPE_NUM_STEPS", 10_000))

    print(
        f"[NPE] device={dev}, sess_per_step={sess_per_step}, "
        f"num_steps={num_steps}, lr={lr}, val={'on' if use_val else 'off'}"
    )

    ema_beta = float(getattr(cfg, "NPE_EMA_BETA", 0.98))
    ema = None

    best_val = float("inf")
    bad_val = 0
    best_state_dict = None

    sample_gen = torch.Generator(device=dev)
    sample_gen.manual_seed(int(seed) + 1)

    def _compute_val_loss() -> float:
        density_estimator.eval()
        with torch.no_grad():
            n = theta_val.shape[0]
            total = 0.0
            count = 0
            for s in range(0, n, val_batch):
                e = min(n, s + val_batch)
                losses = density_estimator.loss(theta_val[s:e], condition=x_val[s:e])
                total += float(losses.sum().item())
                count += int(e - s)
        density_estimator.train()
        return total / max(1, count)

    for step in range(num_steps):
        idx = torch.randint(0, N_train, (sess_per_step,), device=dev, generator=sample_gen)
        theta_b = theta_train.index_select(0, idx)
        x_b = x_train.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        losses = density_estimator.loss(theta_b, condition=x_b)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

        li = float(loss.item())
        ema = li if ema is None else ema_beta * ema + (1.0 - ema_beta) * li

        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: loss={li:.4f} ema={ema:.4f}")

        if use_val and (step + 1) % val_every == 0:
            val_loss = _compute_val_loss()
            improved = val_loss < best_val - val_min_delta
            marker = "*" if improved else " "
            print(
                f"[VAL] step {step + 1}: val_loss={val_loss:.4f} "
                f"(best={best_val:.4f}) {marker}"
            )
            if improved:
                best_val = val_loss
                bad_val = 0
                best_state_dict = {
                    k: v.detach().clone() for k, v in density_estimator.state_dict().items()
                }
            else:
                bad_val += 1
                if bad_val >= val_patience:
                    print(
                        f"[NPE] early stop at step {step + 1} "
                        f"(no val improvement for {val_patience} evals)"
                    )
                    break

        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and (step + 1) % int(checkpoint_every) == 0
        ):
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(
                {
                    "state_dict": density_estimator.state_dict(),
                    "config": cfg,
                    "step": step + 1,
                    "best_val": best_val if use_val else None,
                },
                checkpoint_path,
            )
            print(f"[NPE] checkpoint saved at step {step + 1}: {checkpoint_path}")

    if use_val and best_state_dict is not None:
        density_estimator.load_state_dict(best_state_dict, strict=True)
        print(f"[NPE] Restored best-val weights (val_loss={best_val:.4f})")

    density_estimator.eval()
    posterior = DirectPosterior(density_estimator, prior_theta)
    return density_estimator, posterior


def run_inference_npe(cfg, inference_obj, density_estimator, x_o_flat, prior_theta):
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


def load_npe(
    model_path: str,
    *,
    prior_theta,
    device: str = "cpu",
):
    """Rebuild the NPE architecture from a saved config and load its weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint["config"]

    dev = torch.device(device)

    P = max_num_pulses()
    T = int(saved_cfg.NUM_TRIALS_OBS)
    ar = bool(getattr(saved_cfg, "AUTOREGRESSIVE", False))
    trial_dim = trial_feature_dim(P, ar=ar)
    x_dim = T * trial_dim

    theta_probe = torch.as_tensor(
        prior_theta.sample((1,)),
        device=dev,
        dtype=torch.float32,
    ).reshape(-1)
    theta_dim = int(theta_probe.numel())

    embedding_net = _build_npe_embedding_net(saved_cfg, T=T, P=P, autoregressive=ar)
    est_builder = _build_npe_estimator_builder(saved_cfg, embedding_net)

    dummy_theta = torch.randn(2, theta_dim, device=dev)
    dummy_x = torch.randn(2, x_dim, device=dev)
    density_estimator = est_builder(dummy_theta, dummy_x)

    density_estimator.load_state_dict(checkpoint["state_dict"], strict=True)
    density_estimator.to(dev)
    density_estimator.eval()

    return density_estimator, saved_cfg


def load_npe_decoupled(
    model_path: str,
    *,
    prior_theta,
    device: str = "cpu",
):
    """Load NPE and detach the embedding net so callers can feed variable-length x.

    Returns (density_estimator_with_identity_embedding, embedding_net, saved_cfg, T).
    """
    de, saved_cfg = load_npe(model_path, prior_theta=prior_theta, device=device)
    T = int(saved_cfg.NUM_TRIALS_OBS)
    embedding_net = de.net._embedding_net
    emb_out_dim = int(saved_cfg.NPE_EMBEDDING_OUTPUT_DIM)
    de.net._embedding_net = torch.nn.Identity()
    de._condition_shape = torch.Size([emb_out_dim])
    return de, embedding_net, saved_cfg, T


def _compute_ranks(theta_true: torch.Tensor, posterior_samples: torch.Tensor) -> torch.Tensor:
    theta_true = theta_true.reshape(-1)
    return (posterior_samples < theta_true[None, :]).sum(dim=0).to(torch.int64)


def _plot_sbc_rank_histograms(
    ranks: np.ndarray,
    *,
    param_names: Optional[Sequence[str]] = None,
    outpath: Optional[str] = None,
    bins: int = 30,
):
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
    ar = bool(getattr(cfg, "AUTOREGRESSIVE", False))
    trial_dim = trial_feature_dim(P, ar=ar)
    session_fn = _pick_session_simulator(ar)

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

        _, x_o = session_fn(
            prior_theta,
            num_sessions=1,
            num_trials=T,
            simulate_batch_fn=simulate_batch_fn,
            device=dev,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=_p_success_training_values(cfg),
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
