#!/usr/bin/env python
"""
Train a single amortized NPE model and run inference on marmoset data
across multiple stage conditions.
"""
from __future__ import annotations

import os
import time
import sys
import numpy as np
import torch
import matplotlib
from dataclasses import replace

matplotlib.use("Agg")  
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.inference.posteriors import DirectPosterior
 
from sbi_for_diffusion_models.priors import build_prior_theta_lapse, build_prior_theta
from sbi_for_diffusion_models.models.lapse_rt_choice_model import (
    simulate_rt_choice_batch_lapse,
)
from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    max_num_pulses,
)
from sbi_for_diffusion_models.load_marmoset import load_marmoset_sessions
from sbi_for_diffusion_models.mnpe import (
    train_npe_session,
    load_npe,
    run_sbc_npe,
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

# Configuration
ANIMAL = os.environ.get("ANIMAL", "Aayla")
MODEL_NAME = os.environ.get("MODEL_NAME", "lapse")
N_POST = int(os.environ.get("N_POST", "10000"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.environ.get("OUTDIR", f"marmoset_outputs/{ANIMAL}")
MODEL_DIR = "/projectnb/ssmsvi/jfourie/SBI-trainLapse-4-6/models"
DATA_PATH = "/projectnb/ssmsvi/jfourie/SBI-trainLapse-4-6/marmoset_data.csv.gz"
SKIP_TRAIN=1
DO_SBC=0

# stage P_SUCCESS configs
STAGE_CONFIG = {
    "60-40": 0.6,
    "70-30": 0.7,
    "80-20": 0.8,
}
V_PARAM_INDEX = 2

cfg = RUN_CONFIG_PARAMS

def _select_model(model_name: str):
    """Return (simulate_fn, prior, param_names, tag) for the chosen model."""
    if model_name == "base":
        return (
            simulate_rt_choice_batch,
            build_prior_theta(),
            ("a0", "lam", "v", "B", "tau"),
            "base",
        )
    elif model_name == "lapse":
        return (
            simulate_rt_choice_batch_lapse,
            build_prior_theta_lapse(),
            ("a0", "lam", "v", "B", "tau", "p_lapse"),
            "lapse",
        )
    else:
        raise ValueError(f"Unknown MODEL_NAME={model_name!r}. Use 'base' or 'lapse'.")

def _model_path_for_stage(model_tag: str, stage: str) -> str:
    """Return the checkpoint path for a given model tag and stage."""
    stage_suffix = stage.replace("-", "_")
    return os.path.join(MODEL_DIR, f"npe_{model_tag}_{stage_suffix}.pt")
 
 
def _make_cfg_for_stage(stage: str):
    """Create a RunConfig with P_SUCCESS matching the stage condition."""
    p_success = STAGE_CONFIG[stage]
    return replace(RUN_CONFIG_PARAMS, P_SUCCESS=p_success)
 
# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: Train
# ═══════════════════════════════════════════════════════════════════════════

def train_stage(stage: str, device: str) -> str:
    """Train an NPE model for one stage condition. Returns checkpoint path."""
    simulate_batch_fn, prior_theta, param_names, model_tag = _select_model(MODEL_NAME)
    dev = torch.device(device)
    cfg = _make_cfg_for_stage(stage)
 
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)
 
    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
 
    print(f"\n{'='*60}")
    print(f"  TRAINING  |  stage={stage}  P_SUCCESS={cfg.P_SUCCESS}")
    print(f"  model={model_tag}  T={T}  P={P}")
    print(f"  reservoir={getattr(cfg, 'NPE_RESERVOIR_SIZE', cfg.NPE_NUM_SESSIONS)}")
    print(f"  steps={cfg.NPE_NUM_STEPS}  batch={cfg.NPE_SESSIONS_PER_STEP}")
    print(f"{'='*60}\n")
 
    t0 = time.time()
    density_estimator, posterior = train_npe_session(
        cfg,
        prior_theta,
        simulate_batch_fn=simulate_batch_fn,
        device=device,
        seed=SEED,
    )
    elapsed = time.time() - t0
    print(f"\n[TRAIN] stage={stage} finished in {elapsed / 60:.1f} min")
 
    # Save checkpoint (config with correct P_SUCCESS is embedded)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = _model_path_for_stage(model_tag, stage)
    torch.save(
        {"state_dict": density_estimator.state_dict(), "config": cfg},
        model_path,
    )
    print(f"[TRAIN] Saved: {model_path}")
 
    # SBC
    if DO_SBC:
        print(f"\n--- SBC for stage={stage} ---")
        sbc_dir = os.path.join(OUTDIR, "plots", "sbc", stage.replace("-", "_"))
        run_sbc_npe(
            cfg,
            prior_theta=prior_theta,
            posterior=posterior,
            simulate_batch_fn=simulate_batch_fn,
            device=device,
            num_datasets=int(cfg.NPE_SBC_NUM_DATASETS),
            posterior_samples_per_dataset=int(cfg.NPE_SBC_POST_SAMPLES),
            seed=SEED,
            param_names=param_names,
            outdir=sbc_dir,
            plot_bins=30,
        )
 
    return model_path

# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2: Inference on marmoset data
# ═══════════════════════════════════════════════════════════════════════════

def infer_stage(
    model_path: str,
    animal: str,
    stage: str,
    device: str,
) -> list[torch.Tensor]:
    """
    Run posterior inference on marmoset data for one animal × stage.
    Returns list of posterior sample tensors (one per session).
    """
    simulate_batch_fn, prior_theta, param_names, model_tag = _select_model(MODEL_NAME)
    dev = torch.device(device)
 
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)
 
    density_estimator, saved_cfg = load_npe(
        model_path, prior_theta=prior_theta, device=device,
    )
    posterior = DirectPosterior(density_estimator, prior_theta)
 
    T = int(saved_cfg.NUM_TRIALS_OBS)
    P = max_num_pulses()
 
    print(f"\n{'='*60}")
    print(f"  INFERENCE  |  animal={animal}  stage={stage}  "
          f"P_SUCCESS={saved_cfg.P_SUCCESS}")
    print(f"{'='*60}")
 
    # Check data file exists
    print(f"  Data file: {DATA_PATH}")
    print(f"  Exists: {os.path.isfile(DATA_PATH)}")
    try:
        import pandas as pd
        df_check = pd.read_csv(
            DATA_PATH,
            compression="gzip", 
            nrows=5,
        )
        print(f"  Columns: {list(df_check.columns)}")
        print(f"  Preview:\n{df_check.head(2)}")
    except Exception as e:
        print(f"  [DIAG] Failed to read file: {e}")
 
    try:
        sessions, session_meta = load_marmoset_sessions(
            DATA_PATH,
            animal=animal,
            stage=stage,
            num_trials_per_session=T,
            log_rt=bool(saved_cfg.LOG_RT_MANUALLY),
            seed=SEED,
        )
    except Exception as e:
        print(f"[WARN] Skipping {animal}/{stage}: {e}")
        return []
 
    if len(sessions) == 0:
        print(f"[WARN] No valid sessions for {animal}/{stage}")
        return []
    
    # Pad or truncate sessions to match model's expected input width
    x_dim_expected = T * (2 +P)
    padded_sessions = []
    valid_meta = []
    for s, m in zip(sessions, session_meta):
        x_width = s.shape[1]
        if x_width == x_dim_expected:
            padded_sessions.append(s)
            valid_meta.append(m)
        elif x_width < x_dim_expected:
            pad = torch.zeros(1, x_dim_expected - x_width, dtype=s.dtype)
            padded_sessions.append(torch.cat([s, pad], dim=1))
            valid_meta.append(m)
            print(f"    [PAD] Session {m['session_datetime']}: "
                  f"{m['n_trials']} trials → padded to {T}")
        else:
            padded_sessions.append(s[:, :x_dim_expected])
            valid_meta.append(m)
    sessions = padded_sessions
    session_meta = valid_meta
    
    stage_outdir = os.path.join(OUTDIR, f"{animal}_{stage.replace('-', '_')}")
    plots_dir = os.path.join(stage_outdir, "plots")
    os.makedirs(stage_outdir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
 
    all_samples = []
    for s_idx, (x_o_flat, meta) in enumerate(zip(sessions, session_meta)):
        sess_dt = meta["session_datetime"]
        n_trials = meta["n_trials"]
        acc = meta["accuracy"]
        med_rt = meta["rt_median"]
 
        print(f"\n  Session {s_idx}: {sess_dt}  n={n_trials}  "
              f"acc={acc:.3f}  median_rt={med_rt:.2f}s")
 
        x_o_flat = x_o_flat.to(dev, dtype=torch.float32)
 
        t0 = time.time()
        samples = posterior.sample(
            (N_POST,), x=x_o_flat, show_progress_bars=False,
        ).detach().cpu()
        elapsed = time.time() - t0
        print(f"    Sampled {N_POST} posteriors in {elapsed:.1f}s")
 
        # Save samples
        npy_path = os.path.join(stage_outdir, f"posterior_session_{s_idx}.npy")
        np.save(npy_path, samples.numpy())
 
        # Pairplot
        fig, _ = pairplot(
            samples,
            labels=list(param_names),
            figsize=(10, 10),
        )
        fig.suptitle(
            f"{animal} | {stage} | session {s_idx}\n"
            f"{sess_dt}  n={n_trials}  acc={acc:.3f}  med_rt={med_rt:.2f}s",
            fontsize=10,
        )
        fig_path = os.path.join(plots_dir, f"pairplot_session_{s_idx}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {npy_path}")
        print(f"    Saved: {fig_path}")
 
        all_samples.append(samples)
 
        # Posterior summary
        means = samples.mean(dim=0)
        stds = samples.std(dim=0)
        print("    Posterior means ± std:")
        for name, m, s in zip(param_names, means, stds):
            print(f"      {name:>8s}: {m:.4f} ± {s:.4f}")
 
    # Cross-session summary
    if len(all_samples) > 1:
        _plot_cross_session_summary(
            all_samples, session_meta, param_names, animal, stage, stage_outdir,
        )
 
    print(f"\n[INFERENCE] {animal}/{stage}: {len(sessions)} sessions processed.")
    return all_samples
 

def plot_psuccess_vs_drift(
    stage_results: dict[str, list[torch.Tensor]],
    param_names: tuple[str, ...],
    animal: str,
):
    """
    Plot generative probability (P_SUCCESS) vs learned drift rate (v).
 
    For each stage, aggregates posterior samples across all sessions,
    computes mean and std of v, and plots with error bars.
    """
    stages_sorted = sorted(stage_results.keys(), key=lambda s: STAGE_CONFIG[s])
 
    p_success_vals = []
    v_means = []
    v_stds = []
    v_session_means = {}
 
    for stage in stages_sorted:
        p_succ = STAGE_CONFIG[stage]
        samples_list = stage_results[stage]
 
        if len(samples_list) == 0:
            continue
 
        # Aggregate v across all sessions for this stage
        all_v = torch.cat(
            [s[:, V_PARAM_INDEX] for s in samples_list], dim=0
        )
        p_success_vals.append(p_succ)
        v_means.append(all_v.mean().item())
        v_stds.append(all_v.std().item())
 
        # Per-session means for scatter
        v_session_means[stage] = [
            s[:, V_PARAM_INDEX].mean().item() for s in samples_list
        ]
 
    if len(p_success_vals) == 0:
        print("[WARN] No data for p_success vs drift plot")
        return
 
    p_success_vals = np.array(p_success_vals)
    v_means = np.array(v_means)
    v_stds = np.array(v_stds)
 
    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
 
    # Aggregate mean ± std
    ax.errorbar(
        p_success_vals, v_means, yerr=v_stds,
        fmt="s-", color="black", markersize=10, capsize=6,
        linewidth=2, label="mean ± std (all sessions)",
        zorder=3,
    )
 
    # Individual session means as scatter
    colors = {"60-40": "#2196F3", "70-30": "#4CAF50", "80-20": "#FF9800"}
    for stage in stages_sorted:
        if stage not in v_session_means:
            continue
        p_succ = STAGE_CONFIG[stage]
        sess_vs = v_session_means[stage]
        ax.scatter(
            [p_succ] * len(sess_vs), sess_vs,
            color=colors.get(stage, "gray"),
            alpha=0.5, s=40, edgecolors="white", linewidth=0.5,
            label=f"{stage} sessions (n={len(sess_vs)})",
            zorder=2,
        )
 
    ax.set_xlabel("Generative probability (P_SUCCESS)", fontsize=13)
    ax.set_ylabel("Learned drift rate (v)", fontsize=13)
    ax.set_title(
        f"{animal} — P_SUCCESS vs posterior drift rate",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(sorted(STAGE_CONFIG.values()))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    plots_dir = os.path.join(OUTDIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, "psuccess_vs_drift_rate.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SUMMARY] Saved: {fig_path}")
 
    # Print table
    print(f"\n  {'Stage':>8s}  {'P_SUCCESS':>10s}  {'v (mean)':>10s}  {'v (std)':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
    for stage, p, vm, vs in zip(stages_sorted, p_success_vals, v_means, v_stds):
        print(f"  {stage:>8s}  {p:>10.2f}  {vm:>10.4f}  {vs:>10.4f}")
 
 

def _plot_cross_session_summary(
    all_samples, session_meta, param_names, animal, stage, outdir,
):
    """Plot posterior means ± std across sessions for each parameter."""
    n_sessions = len(all_samples)
    D = all_samples[0].shape[1]
 
    means = np.array([s.mean(dim=0).numpy() for s in all_samples])
    stds = np.array([s.std(dim=0).numpy() for s in all_samples])
 
    fig, axes = plt.subplots(D, 1, figsize=(8, 2.5 * D), constrained_layout=True)
    if D == 1:
        axes = [axes]
 
    x_pos = np.arange(n_sessions)
    for d, ax in enumerate(axes):
        ax.errorbar(x_pos, means[:, d], yerr=stds[:, d], fmt="o-", capsize=3)
        ax.set_ylabel(param_names[d])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [m["session_datetime"][:10] for m in session_meta],
            rotation=45, ha="right", fontsize=7,
        )
 
    axes[0].set_title(f"{animal} | {stage} — posterior across sessions")
    axes[-1].set_xlabel("session")
 
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, "cross_session_summary.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_path}")
 


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, param_names, model_tag = _select_model(MODEL_NAME)
 
    print(f"Device:  {device}")
    print(f"Animal:  {ANIMAL}")
    print(f"Model:   {MODEL_NAME} ({model_tag})")
    print(f"Stages:  {list(STAGE_CONFIG.keys())}")
    print(f"Data:    {DATA_PATH}")
    print(f"Models:  {MODEL_DIR}")
    print(f"Output:  {OUTDIR}")

    # ── Phase 1: Train one model per stage ─────────────────────────────────
    model_paths = {}
    for stage in STAGE_CONFIG:
        expected_path = _model_path_for_stage(model_tag, stage)
 
        if SKIP_TRAIN:
            if os.path.isfile(expected_path):
                model_paths[stage] = expected_path
                print(f"\n[SKIP_TRAIN] {stage}: using {expected_path}")
            else:
                print(f"\n[WARN] {stage}: model not found at {expected_path}, "
                      f"will train")
                model_paths[stage] = train_stage(stage, device)
        else:
            model_paths[stage] = train_stage(stage, device)
 
    # ── Phase 2: Inference per stage ───────────────────────────────────────
    stage_results: dict[str, list[torch.Tensor]] = {}
 
    for stage in STAGE_CONFIG:
        if stage not in model_paths:
            continue
        samples = infer_stage(
            model_paths[stage], ANIMAL, stage, device,
        )
        stage_results[stage] = samples
 
    # ── Phase 3: P_SUCCESS vs drift rate summary plot ──────────────────────
    plot_psuccess_vs_drift(stage_results, param_names, ANIMAL)
 
    print(f"\n{'='*60}")
    print(f"  ALL DONE  |  outputs in {OUTDIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()