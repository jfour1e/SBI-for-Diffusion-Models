"""Prior predictive check for RT-choice models.

Draws N sessions from the prior, simulates behavioral data, and writes:
  1. prior marginal parameter distributions
  2. pooled RT histogram (non-timeout)
  3. per-session choice accuracy
  4. per-session timeout rate
  5. per-session median RT

Environment overrides: MODEL_NAME ("base"|"lapse"), N_SESSIONS, SEED, OUTDIR.
"""
from __future__ import annotations

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi_for_diffusion_models.priors import build_prior_theta, build_prior_theta_lapse
from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    max_num_pulses,
)
from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
)
from sbi_for_diffusion_models.data_simulator import simulate_training_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, T_MAX

cfg = RUN_CONFIG_PARAMS

MODEL_NAME = os.environ.get("MODEL_NAME", "lapse")
N_SESSIONS = int(os.environ.get("N_SESSIONS", "100"))
SEED       = int(os.environ.get("SEED", "42"))
OUTDIR     = os.environ.get("OUTDIR", "prior_predictive_outputs")


def get_model_spec(model_name: str) -> dict:
    if model_name == "base":
        return {
            "prior_builder": build_prior_theta,
            "simulate_batch_fn": simulate_rt_choice_batch,
            "param_names": ["a0", "lam", "v", "B", "tau"],
        }
    if model_name == "lapse":
        return {
            "prior_builder": build_prior_theta_lapse,
            "simulate_batch_fn": simulate_rt_choice_batch_lapse,
            "param_names": ["a0", "lam", "v", "B", "tau", "p_lapse"],
        }
    raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use 'base' or 'lapse'.")


def unpack_sessions(
    x_all: torch.Tensor,
    theta_all: torch.Tensor,
    T: int,
    P: int,
    log_rt: bool,
) -> dict:
    N = x_all.shape[0]
    trial_dim = 2 + P
    x_3d = x_all.view(N, T, trial_dim)

    rt     = x_3d[:, :, 0].cpu().numpy()
    choice = x_3d[:, :, 1].cpu().numpy()
    pulses = x_3d[:, :, 2:].cpu().numpy()

    rt_real = np.exp(rt) if log_rt else rt

    pulse_sum    = pulses.sum(axis=-1)
    correct_side = (pulse_sum > 0).astype(float)
    correct_all  = (choice == correct_side)

    log_tmax = math.log(T_MAX) if log_rt else T_MAX
    hit_all  = rt < log_tmax - 1e-4

    return {
        "rt_all":     rt_real,
        "rt_stored":  rt,
        "choice_all": choice,
        "pulses_all": pulses,
        "theta_all":  theta_all.cpu().numpy(),
        "correct_all": correct_all,
        "hit_all":    hit_all,
    }


def plot_prior_predictive(data: dict, param_names: list[str], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    rt_real = data["rt_all"]
    correct = data["correct_all"]
    hit     = data["hit_all"]
    theta   = data["theta_all"]
    N, T    = rt_real.shape
    D       = theta.shape[1]

    ncols = min(D, 4)
    nrows = math.ceil(D / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()
    for d, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(theta[:, d], bins=40, density=True, color="steelblue", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle("Prior Marginal Distributions", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "1_prior_marginals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    rt_flat = rt_real[hit]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(rt_flat, bins=80, density=True, color="coral", alpha=0.8)
    axes[0].set_xlabel("RT (s)")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Marginal RT  (n={len(rt_flat):,})")
    axes[0].axvline(np.median(rt_flat), color="k", linestyle="--",
                    label=f"median={np.median(rt_flat):.2f}s")
    axes[0].legend()

    axes[1].hist(np.log(rt_flat + 1e-9), bins=80, density=True, color="coral", alpha=0.8)
    axes[1].set_xlabel("log RT")
    axes[1].set_ylabel("density")
    axes[1].set_title("Marginal log(RT)")
    fig.suptitle("Prior Predictive RT Distribution", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, "2_rt_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    acc_per_session = np.array([
        correct[i][hit[i]].mean() if hit[i].sum() > 0 else np.nan
        for i in range(N)
    ])
    acc_valid = acc_per_session[~np.isnan(acc_per_session)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(acc_valid, bins=40, density=True, color="mediumseagreen", alpha=0.8)
    ax.axvline(0.5, color="k", linestyle="--", label="chance")
    ax.axvline(acc_valid.mean(), color="red", linestyle="-",
               label=f"mean={acc_valid.mean():.2f}")
    ax.set_xlabel("Proportion correct")
    ax.set_ylabel("density")
    ax.set_title(f"Prior Predictive Choice Accuracy  (N={N} sessions)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "3_choice_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    timeout_rate = (~hit).mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(timeout_rate, bins=40, density=True, color="goldenrod", alpha=0.8)
    ax.axvline(timeout_rate.mean(), color="red", linestyle="-",
               label=f"mean={timeout_rate.mean():.2%}")
    ax.set_xlabel("Timeout fraction per session")
    ax.set_ylabel("density")
    ax.set_title(f"Prior Predictive Timeout Rate  (N={N} sessions)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "4_timeout_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    med_rt = np.array([
        np.median(rt_real[i][hit[i]]) if hit[i].sum() > 0 else np.nan
        for i in range(N)
    ])
    med_rt_valid = med_rt[~np.isnan(med_rt)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(med_rt_valid, bins=40, density=True, color="mediumpurple", alpha=0.8)
    ax.set_xlabel("Median RT per session (s)")
    ax.set_ylabel("density")
    ax.set_title("Prior Predictive: Per-Session Median RT")
    fig.tight_layout()
    path = os.path.join(outdir, "5_median_rt_per_session.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Prior Predictive Check Summary\n")
        f.write("==============================\n")
        f.write(f"Model:       {MODEL_NAME}\n")
        f.write(f"N sessions:  {N}\n")
        f.write(f"T trials:    {T}\n\n")
        f.write(
            f"RT (non-timeout): mean={rt_flat.mean():.3f}s  "
            f"median={np.median(rt_flat):.3f}s  "
            f"5th={np.percentile(rt_flat,5):.3f}s  "
            f"95th={np.percentile(rt_flat,95):.3f}s\n"
        )
        f.write(f"Choice accuracy:  mean={acc_valid.mean():.3f}  std={acc_valid.std():.3f}\n")
        f.write(f"Timeout rate:     mean={timeout_rate.mean():.3f}  max={timeout_rate.max():.3f}\n")
        for d, name in enumerate(param_names):
            col = theta[:, d]
            f.write(
                f"  {name:10s}: mean={col.mean():.3f}  std={col.std():.3f}  "
                f"[{col.min():.3f}, {col.max():.3f}]\n"
            )
    print("Saved:", summary_path)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev    = torch.device(device)
    print(f"Model: {MODEL_NAME}  |  N_sessions: {N_SESSIONS}  |  device: {device}")

    spec              = get_model_spec(MODEL_NAME)
    prior_theta       = spec["prior_builder"]()
    simulate_batch_fn = spec["simulate_batch_fn"]
    param_names       = spec["param_names"]

    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
    print(f"P={P}, T={T}")

    print(f"\nSimulating {N_SESSIONS} sessions from prior ...")
    theta_all, x_all = simulate_training_sessions(
        prior_theta=prior_theta,
        num_sessions=N_SESSIONS,
        num_trials=T,
        simulate_batch_fn=simulate_batch_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=SEED,
    )

    data = unpack_sessions(x_all, theta_all, T, P, log_rt=bool(cfg.LOG_RT_MANUALLY))

    rt_flat = data["rt_all"][data["hit_all"]]
    print(
        f"\nRT summary (non-timeout):  "
        f"mean={rt_flat.mean():.3f}s  median={np.median(rt_flat):.3f}s  "
        f"[{rt_flat.min():.3f}, {rt_flat.max():.3f}]"
    )
    print(f"Accuracy:  mean={data['correct_all'][data['hit_all']].mean():.3f}")
    print(f"Timeouts:  {(~data['hit_all']).mean():.2%} of all trials")

    print(f"\nPlotting to {OUTDIR}/ ...")
    plot_prior_predictive(data, param_names, OUTDIR)


if __name__ == "__main__":
    main()
