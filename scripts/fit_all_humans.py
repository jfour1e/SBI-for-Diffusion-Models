"""Fit the trained human NPE model to every human subject in the dataset.

Outputs (default OUTDIR=group_outputs_humans)
---------------------------------------------
  manifest.json
  all_subjects.csv           # tidy: one row per subject
  group_pairwise_tests.csv   # pairwise t-test + MWU per (parameter, group-pair)
  group_anova.csv            # 1-way ANOVA per parameter (3 groups)
  group_violins.png          # violin plot per group per param  (line-plot analogue)
  group_box_strip.png        # boxplot + strip per group per param
  subjects/<name>/
      posterior.npy
      summary.txt
      ppc.png                 # one PPC per subject (if PPC=1)

Usage
-----
    MODEL_NAME=lapse_noleak_ar_human python scripts/fit_all_humans.py
"""
from __future__ import annotations

import os
import json
import math
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.model_specs import select_model, simulation_overrides_for
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions_ar,
    trial_feature_dim,
)
from sbi_for_diffusion_models.mnpe import load_npe_decoupled
from sbi_for_diffusion_models.load_human import load_human_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, T_MAX

cfg = RUN_CONFIG_PARAMS

N_POST     = int(os.environ.get("N_POST", "5000"))
N_PPC      = int(os.environ.get("N_PPC",  "200"))
SEED       = int(os.environ.get("SEED",   "0"))
OUTDIR     = os.environ.get("OUTDIR",     "group_outputs_humans")
MODEL_DIR  = os.path.expanduser(os.environ.get("MODEL_DIR", "models"))
MODEL_NAME = os.environ.get("MODEL_NAME", "lapse_noleak_ar_human")
DATA_PATH  = os.environ.get(
    "DATA_PATH", "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/ALLspecies_trials_combined.csv"
)
DO_PPC = int(os.environ.get("PPC", "1"))


def _default_model_file(model_name: str, cfg) -> str:
    values = "_".join(f"{int(round(v * 100)):03d}" for v in cfg.P_SUCCESS_TRAIN_VALUES)
    return f"npe_{model_name}_mixed_{values}.pt"


MODEL_FILE = os.environ.get("MODEL_FILE", _default_model_file(MODEL_NAME, cfg))


_, _, PARAM_NAMES_TUPLE, _MODEL_TAG, AUTOREGRESSIVE = select_model(MODEL_NAME)
PARAM_NAMES = list(PARAM_NAMES_TUPLE)


# ---------------------------------------------------------------------------
# PPC
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_from_posterior_human(posterior_samples: np.ndarray, n_ppc: int,
                                  T: int, P: int, device: str = "cpu") -> dict:
    """Forward-simulate from posterior samples using the human simulator+cascade."""
    dev = torch.device(device)
    replace = len(posterior_samples) < n_ppc
    idx = np.random.choice(len(posterior_samples), size=n_ppc, replace=replace)
    theta = torch.tensor(posterior_samples[idx], dtype=torch.float32, device=dev)

    log_tmax = math.log(T_MAX) if cfg.LOG_RT_MANUALLY else T_MAX
    trial_dim_ar = trial_feature_dim(P, ar=True)

    sim_fn, _prior, _, _, _ = select_model(MODEL_NAME)
    overrides = simulation_overrides_for(MODEL_NAME)
    pulse_gen = overrides.get("pulse_generator_fn")
    p_sampler = overrides.get("p_success_per_trial_fn")

    _, x_flat = simulate_training_sessions_ar(
        prior_theta=None,
        num_sessions=n_ppc,
        num_trials=T,
        simulate_batch_fn=sim_fn,
        device=dev,
        mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(cfg.P_SUCCESS),  # ignored when cascade sampler is set
        P=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(np.random.randint(0, 2**31 - 1)),
        theta=theta,
        warn_on_timeouts=False,
        pulse_generator_fn=pulse_gen,
        p_success_per_trial_fn=p_sampler,
    )
    x_3d = x_flat.view(n_ppc, T, trial_dim_ar).cpu().numpy()

    rt = np.exp(x_3d[..., 0]) if cfg.LOG_RT_MANUALLY else x_3d[..., 0]
    choice = x_3d[..., 1].astype(np.int32)
    pulses = x_3d[..., 4:]
    hit = x_3d[..., 0] < log_tmax - 1e-4
    correct_side = (pulses.sum(axis=-1) > 0).astype(np.int32)

    acc_per_draw = np.array([
        (choice[i][hit[i]] == correct_side[i][hit[i]]).mean() if hit[i].any() else np.nan
        for i in range(n_ppc)
    ])
    p_right_per_draw = np.array([
        choice[i][hit[i]].mean() if hit[i].any() else np.nan
        for i in range(n_ppc)
    ])
    timeout_per_draw = 1.0 - hit.mean(axis=1)
    return {
        "rt": rt.astype(np.float32),
        "choice": choice,
        "hit": hit,
        "acc_per_draw": acc_per_draw,
        "p_right_per_draw": p_right_per_draw,
        "timeout_per_draw": timeout_per_draw,
    }


def _extract_real(x_flat: torch.Tensor, P: int) -> dict:
    trial_dim = trial_feature_dim(P, ar=AUTOREGRESSIVE)
    T = x_flat.shape[1] // trial_dim
    x = x_flat.reshape(T, trial_dim).numpy()
    log_rt = x[:, 0]
    choice = x[:, 1].astype(int)
    pulses = x[:, 4:] if AUTOREGRESSIVE else x[:, 2:]
    rt = np.exp(log_rt) if cfg.LOG_RT_MANUALLY else log_rt
    hit = rt < float(T_MAX) - 0.01
    pulse_sum = pulses.sum(axis=1)
    correct_side = (pulse_sum > 0).astype(int)
    return {
        "rt": rt, "choice": choice, "hit": hit,
        "correct_side": correct_side, "n_trials": T,
    }


def plot_ppc(real: dict, sim: dict, label: str, outpath: str, n_ppc: int):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    real_hit = real["hit"]
    real_acc = (real["choice"][real_hit] == real["correct_side"][real_hit]).mean() \
        if real_hit.any() else float("nan")
    fig.suptitle(f"PPC — {label}  (real acc={real_acc:.1%})", fontsize=12)

    bins = np.linspace(0, float(T_MAX), 41)
    real_rt_hit = real["rt"][real["hit"]]
    sim_rt_hit  = sim["rt"].reshape(-1)[sim["hit"].reshape(-1)]

    ax = axes[0]
    ax.hist(real_rt_hit, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="Real")
    ax.hist(sim_rt_hit, bins=bins, density=True, alpha=0.4,
            color="tomato", label=f"PPC (n={n_ppc})")
    ax.set_xlabel("RT (s)"); ax.set_ylabel("Density")
    ax.set_title("RT distribution"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(sim["acc_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_acc, color="steelblue", lw=2, label=f"Real ({real_acc:.2f})")
    ax.set_xlabel("Accuracy"); ax.set_title("Accuracy"); ax.legend(fontsize=8)

    ax = axes[2]
    real_p_right = real["choice"][real_hit].mean() if real_hit.any() else np.nan
    ax.hist(sim["p_right_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_p_right, color="steelblue", lw=2, label=f"Real ({real_p_right:.2f})")
    ax.set_xlabel("P(right)"); ax.set_title("Side preference"); ax.legend(fontsize=8)

    ax = axes[3]
    real_timeout = 1.0 - real_hit.mean()
    ax.hist(sim["timeout_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_timeout, color="steelblue", lw=2, label=f"Real ({real_timeout:.2f})")
    ax.set_xlabel("Timeout rate"); ax.set_title("Timeout"); ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-subject fit
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    subject: str
    group: str
    n_trials: int
    posterior_mean: np.ndarray
    posterior_std: np.ndarray


def fit_one_subject(subject: str, group: str,
                    posterior: DirectPosterior, embedding_net: torch.nn.Module,
                    device: str, outdir: str) -> CellResult | None:
    P = max_num_pulses()
    trial_dim = trial_feature_dim(P, ar=AUTOREGRESSIVE)

    try:
        sessions, _meta = load_human_sessions(
            csv_path=DATA_PATH,
            subject=subject,
            autoregressive=AUTOREGRESSIVE,
            test_only=True,
            drop_omissions=True,
        )
    except Exception as e:
        print(f"  [skip] {subject}: {e}")
        return None
    if not sessions:
        print(f"  [skip] {subject}: no test trials after filtering")
        return None

    x_session = sessions[0].to(device, dtype=torch.float32)
    T = x_session.shape[1] // trial_dim

    cell_dir = os.path.join(outdir, "subjects", subject)
    os.makedirs(cell_dir, exist_ok=True)

    with torch.no_grad():
        x_emb = embedding_net(x_session)
    samples = posterior.sample(
        (N_POST,), x=x_emb, show_progress_bars=False,
    ).detach().cpu().numpy()
    np.save(os.path.join(cell_dir, "posterior.npy"), samples)

    mean_vec = samples.mean(axis=0)
    std_vec  = samples.std(axis=0)

    with open(os.path.join(cell_dir, "summary.txt"), "w") as f:
        f.write(f"Subject: {subject}  Group: {group}  n_test_trials: {T}\n\n")
        for d, name in enumerate(PARAM_NAMES):
            f.write(f"  {name:8s}: mean={mean_vec[d]:.4f}  std={std_vec[d]:.4f}  "
                    f"[{np.percentile(samples[:,d],5):.4f}, "
                    f"{np.percentile(samples[:,d],95):.4f}]\n")

    if DO_PPC:
        try:
            real = _extract_real(x_session.cpu(), P)
            sim = simulate_from_posterior_human(samples, N_PPC, T=T, P=P, device="cpu")
            plot_ppc(real, sim,
                     label=f"{subject} ({group}, {T} trials)",
                     outpath=os.path.join(cell_dir, "ppc.png"),
                     n_ppc=N_PPC)
        except Exception:
            print(f"  [warn] PPC failed for {subject}:")
            traceback.print_exc()

    print(f"  [ok]  {subject:14s} group={group:4s} "
          f"n_trials={T:4d} "
          f"means=[" + ", ".join(f"{v:.3f}" for v in mean_vec) + "]")
    return CellResult(subject=subject, group=group, n_trials=T,
                      posterior_mean=mean_vec, posterior_std=std_vec)


# ---------------------------------------------------------------------------
# Group analyses
# ---------------------------------------------------------------------------

def group_pairwise_tests(df: pd.DataFrame, groups=("TD", "ASD", "PMS")) -> pd.DataFrame:
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]
    rows = []
    for a, b in pairs:
        for p in PARAM_NAMES:
            xa = df.loc[df["group"] == a, p].dropna().values
            xb = df.loc[df["group"] == b, p].dropna().values
            row = dict(group_a=a, group_b=b, param=p, n_a=len(xa), n_b=len(xb))
            if len(xa) >= 2 and len(xb) >= 2:
                t_res = stats.ttest_ind(xb, xa, equal_var=False, nan_policy="omit")
                mw = stats.mannwhitneyu(xb, xa, alternative="two-sided")
                pooled_sd = math.sqrt(
                    ((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1))
                    / max(1, (len(xa) + len(xb) - 2))
                )
                row.update(
                    mean_a=float(xa.mean()), mean_b=float(xb.mean()),
                    sd_a=float(xa.std(ddof=1)), sd_b=float(xb.std(ddof=1)),
                    t_stat=float(t_res.statistic), t_p=float(t_res.pvalue),
                    mwu_stat=float(mw.statistic), mwu_p=float(mw.pvalue),
                    cohens_d=float((xb.mean() - xa.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
                )
            else:
                row.update(mean_a=np.nan, mean_b=np.nan, sd_a=np.nan, sd_b=np.nan,
                           t_stat=np.nan, t_p=np.nan, mwu_stat=np.nan, mwu_p=np.nan, cohens_d=np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def group_anova(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PARAM_NAMES:
        sub = df[["group", p]].dropna().rename(columns={p: "y"})
        if sub["group"].nunique() < 2:
            rows.append(dict(param=p, error="insufficient groups"))
            continue
        groups_present = sorted(sub["group"].unique())
        arrays = [sub.loc[sub["group"] == g, "y"].values for g in groups_present]
        try:
            f_stat, f_p = stats.f_oneway(*arrays)
            kw_stat, kw_p = stats.kruskal(*arrays)
            rows.append(dict(param=p,
                             n_groups=len(groups_present),
                             groups=",".join(groups_present),
                             f_stat=float(f_stat), f_p=float(f_p),
                             kw_stat=float(kw_stat), kw_p=float(kw_p)))
        except Exception as e:
            rows.append(dict(param=p, error=f"{e.__class__.__name__}: {e}"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

GROUP_ORDER = ["TD", "ASD", "PMS"]
GROUP_COLORS = {"TD": "steelblue", "ASD": "mediumseagreen", "PMS": "tomato"}


def plot_group_violins(df: pd.DataFrame, tests: pd.DataFrame, outpath: str):
    D = len(PARAM_NAMES)
    ncols = 4
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        positions = []
        data_list = []
        face_colors = []
        labels = []
        for i, g in enumerate(GROUP_ORDER):
            vals = df.loc[df["group"] == g, name].dropna().values
            if len(vals) >= 2:
                positions.append(i)
                data_list.append(vals)
                face_colors.append(GROUP_COLORS[g])
                labels.append(g)
        if data_list:
            parts = ax.violinplot(data_list, positions=positions, widths=0.7,
                                  showmeans=True, showmedians=False)
            for pc, fc in zip(parts["bodies"], face_colors):
                pc.set_facecolor(fc); pc.set_alpha(0.55); pc.set_edgecolor("black")

            # Strip
            rng = np.random.default_rng(0)
            for pos, vals, fc in zip(positions, data_list, face_colors):
                x_j = pos + (rng.random(len(vals)) - 0.5) * 0.18
                ax.scatter(x_j, vals, s=14, color=fc, edgecolors="black",
                           linewidths=0.4, alpha=0.85, zorder=3)

        ax.set_xticks(list(range(len(GROUP_ORDER))))
        ax.set_xticklabels(GROUP_ORDER, fontsize=9)
        ax.set_title(name)

        # Annotate PMS-vs-TD test on each panel (the key comparison)
        row = tests[(tests["param"] == name)
                    & (((tests["group_a"] == "TD") & (tests["group_b"] == "PMS"))
                       | ((tests["group_a"] == "PMS") & (tests["group_b"] == "TD")))]
        if len(row):
            r = row.iloc[0]
            ax.text(0.02, 0.98,
                    f"PMS vs TD\nt p={r['t_p']:.3g}\nMWU p={r['mwu_p']:.3g}\nd={r['cohens_d']:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7,
                              edgecolor="lightgray"))

    for ax in axes[D:]:
        ax.set_visible(False)

    fig.suptitle("Posterior parameter distributions by group (humans)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_group_boxstrip(df: pd.DataFrame, tests: pd.DataFrame, outpath: str):
    D = len(PARAM_NAMES)
    ncols = 4
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    rng = np.random.default_rng(0)

    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        data = [df.loc[df["group"] == g, name].dropna().values for g in GROUP_ORDER]
        bp = ax.boxplot(data, positions=list(range(len(GROUP_ORDER))),
                        widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.35)
        for i, g in enumerate(GROUP_ORDER):
            y = data[i]
            x_j = i + (rng.random(len(y)) - 0.5) * 0.22
            ax.scatter(x_j, y, s=20, color=GROUP_COLORS[g], edgecolors="black",
                       linewidths=0.4, alpha=0.85, zorder=3)
        ax.set_xticks(list(range(len(GROUP_ORDER))))
        ax.set_xticklabels(GROUP_ORDER, fontsize=9)
        ax.set_title(name)

    for ax in axes[D:]:
        ax.set_visible(False)

    fig.suptitle("Posterior parameter means by group (humans)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    os.makedirs(OUTDIR, exist_ok=True)

    print("Reading human roster ...")
    df_all = pd.read_csv(DATA_PATH, compression="infer",
                        dtype={"flashes_left": str, "flashes_right": str},
                        low_memory=False)
    df_all = df_all[df_all["specie"] == "Human"]
    roster = (df_all[["name", "group"]]
              .drop_duplicates()
              .sort_values(["group", "name"])
              .reset_index(drop=True))
    print(f"Found {len(roster)} human subjects across groups: "
          f"{dict(roster['group'].value_counts())}")

    simulate_batch_fn, prior_theta, _param_names, _model_tag, autoregressive = select_model(MODEL_NAME)
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f"\nLoading NPE from {model_path} on {device} (MODEL_NAME={MODEL_NAME}, ar={autoregressive}) ...")
    de, embedding_net, saved_cfg, _T = load_npe_decoupled(
        model_path, prior_theta=prior_theta, device=device,
    )
    if bool(getattr(saved_cfg, "AUTOREGRESSIVE", False)) != autoregressive:
        raise RuntimeError(f"AR-flag mismatch between MODEL_NAME and checkpoint.")
    posterior = DirectPosterior(posterior_estimator=de, prior=prior_theta, device=device)

    results: list[CellResult] = []
    for _, r in roster.iterrows():
        try:
            cr = fit_one_subject(r["name"], r["group"], posterior, embedding_net, device, OUTDIR)
        except Exception:
            print(f"  [err] {r['name']} failed:")
            traceback.print_exc()
            continue
        if cr is not None:
            results.append(cr)

    if not results:
        raise RuntimeError("No subjects produced posterior samples.")

    rows = []
    for cr in results:
        d = dict(subject=cr.subject, group=cr.group, n_trials=cr.n_trials)
        for i, name in enumerate(PARAM_NAMES):
            d[name]         = float(cr.posterior_mean[i])
            d[f"{name}_sd"] = float(cr.posterior_std[i])
        rows.append(d)
    df = pd.DataFrame(rows)

    out_subj = os.path.join(OUTDIR, "all_subjects.csv")
    df.to_csv(out_subj, index=False)
    print(f"\nSaved per-subject means: {out_subj}")

    tests = group_pairwise_tests(df)
    out_tests = os.path.join(OUTDIR, "group_pairwise_tests.csv")
    tests.to_csv(out_tests, index=False)
    print(f"Saved pairwise tests: {out_tests}")

    anova = group_anova(df)
    out_anova = os.path.join(OUTDIR, "group_anova.csv")
    anova.to_csv(out_anova, index=False)
    print(f"Saved one-way ANOVA: {out_anova}")
    print()
    print(anova.to_string(index=False))
    print()

    plot_group_violins(df, tests, os.path.join(OUTDIR, "group_violins.png"))
    print(f"Saved: {OUTDIR}/group_violins.png  (+ .svg)")
    plot_group_boxstrip(df, tests, os.path.join(OUTDIR, "group_box_strip.png"))
    print(f"Saved: {OUTDIR}/group_box_strip.png  (+ .svg)")

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "model_file": MODEL_FILE,
            "n_subjects_fit": len(results),
            "n_post_samples": N_POST,
            "n_ppc_draws": N_PPC,
            "param_names": PARAM_NAMES,
            "groups_present": sorted(df["group"].unique().tolist()),
        }, f, indent=2)

    print(f"\nAll done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
