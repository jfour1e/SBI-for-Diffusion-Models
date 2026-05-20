"""
Fit the pretrained mixed-p_success NPE to every marmoset at every stage,
run posterior predictive checks (default: only on the 70-30 stage), and run
hypothesis tests across both genotype (Shank3 vs WT) and condition (stage).

Outputs (default OUTDIR=group_outputs_all_stages)
-------------------------------------------------
  manifest.json
  all_animals_all_stages.csv         # tidy: one row per (animal, stage)
  per_stage_group_tests.csv          # Welch + Mann-Whitney per (parameter, stage)
  cross_condition_tests.csv          # mixed-effects ANOVA per parameter
  stage_x_group_lines.png            # mean ± SEM trajectory across stages
  stage_x_group_violins.png          # violin per stage x group per parameter
  per_stage/<stage>/
      group_box_strip.png
      group_pairgrid.png
      per_animal_bars.png
  animals/<animal>/<stage>/
      posterior_combined.npy
      summary.txt
      ppc_combined.png               # only for stages in PPC_STAGES (default 70-30)

Usage
-----
    python scripts/fit_all_marmosets_all_stages.py
    OUTDIR=foo N_POST=2000 PPC_STAGES=70-30 STAGES=100-0,90-10,80-20,70-30,60-40,randomProb \
        python scripts/fit_all_marmosets_all_stages.py
"""
from __future__ import annotations

import os
import math
import json
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

from sbi_for_diffusion_models.priors import build_prior_theta_lapse
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.models.lapse_rt_choice_model import simulate_rt_choice_batch_lapse
from sbi_for_diffusion_models.mnpe import load_npe_decoupled
from sbi_for_diffusion_models.load_marmoset import load_marmoset_sessions
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, T_MAX

cfg = RUN_CONFIG_PARAMS

ALL_STAGES = ["100-0", "90-10", "80-20", "70-30", "60-40", "randomProb"]
STAGE_P_SUCCESS = {
    "100-0":      1.0,
    "90-10":      0.9,
    "80-20":      0.8,
    "70-30":      0.7,
    "60-40":      0.6,
    "randomProb": 0.7,
}

STAGES_ENV  = os.environ.get("STAGES", ",".join(ALL_STAGES))
STAGES      = [s.strip() for s in STAGES_ENV.split(",") if s.strip()]
PPC_STAGES  = [s.strip() for s in os.environ.get("PPC_STAGES", "70-30").split(",") if s.strip()]
N_POST      = int(os.environ.get("N_POST", "5000"))
N_PPC       = int(os.environ.get("N_PPC",  "200"))
SEED        = int(os.environ.get("SEED",   "0"))
OUTDIR      = os.environ.get("OUTDIR",     "group_outputs_all_stages")
MODEL_DIR   = os.path.expanduser(os.environ.get("MODEL_DIR", "models"))
MODEL_FILE  = os.environ.get("MODEL_FILE", "npe_lapse_mixed_100_090_080_070_060.pt")
DATA_PATH   = os.environ.get(
    "DATA_PATH", "/projectnb/ssmsvi/rsenne/data_marmoset/marmoset_data.csv.gz"
)
ANIMALS_FILTER = os.environ.get("ANIMALS", "").strip()

PARAM_NAMES = ["a0", "lam", "v", "B", "tau", "p_lapse"]


def _random_pulses(T: int, P: int, p_success: float, dev: torch.device):
    correct_side = (torch.randint(0, 2, (T,), device=dev) * 2 - 1).float()
    is_correct = (torch.rand(T, P, device=dev) < p_success)
    pulses = torch.where(
        is_correct,
        correct_side.unsqueeze(1).expand(T, P),
        -correct_side.unsqueeze(1).expand(T, P),
    )
    return pulses, correct_side


@torch.no_grad()
def simulate_from_posterior(posterior_samples: np.ndarray, n_ppc: int,
                            T: int, P: int, p_success: float,
                            device: str = "cpu") -> dict:
    dev = torch.device(device)
    replace = len(posterior_samples) < n_ppc
    idx = np.random.choice(len(posterior_samples), size=n_ppc, replace=replace)
    theta = torch.tensor(posterior_samples[idx], dtype=torch.float32, device=dev)

    all_rt, all_choice, all_hit, all_correct = [], [], [], []
    for i in range(n_ppc):
        th_i = theta[i].unsqueeze(0).expand(T, -1)
        pulses, correct_side_i = _random_pulses(T, P, p_success, dev)
        x_raw_i, hit_i, _ = simulate_rt_choice_batch_lapse(
            th_i,
            pulse_sides=pulses,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=p_success,
        )
        correct_choice = ((correct_side_i > 0).long()).cpu().numpy().astype(np.int32)
        all_rt.append(x_raw_i[:, 0].cpu().numpy().astype(np.float32))
        all_choice.append(x_raw_i[:, 1].cpu().numpy().astype(np.int32))
        all_hit.append(hit_i.cpu().numpy().astype(bool))
        all_correct.append(correct_choice)

    rt           = np.stack(all_rt)
    choice       = np.stack(all_choice)
    hit          = np.stack(all_hit)
    correct_side = np.stack(all_correct)

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
        "rt": rt, "choice": choice, "hit": hit,
        "acc_per_draw": acc_per_draw,
        "p_right_per_draw": p_right_per_draw,
        "timeout_per_draw": timeout_per_draw,
    }


def _extract_real(x_flat: torch.Tensor, P: int) -> dict:
    trial_dim = 2 + P
    T = x_flat.shape[1] // trial_dim
    x = x_flat.reshape(T, trial_dim).numpy()
    log_rt = x[:, 0]
    choice = x[:, 1].astype(int)
    pulses = x[:, 2:]
    rt = np.exp(log_rt) if cfg.LOG_RT_MANUALLY else log_rt
    hit = rt < float(T_MAX) - 0.01
    pulse_sum = pulses.sum(axis=1)
    correct_side = (pulse_sum > 0).astype(int)
    return {
        "rt": rt, "choice": choice, "hit": hit,
        "correct_side": correct_side, "n_trials": T,
    }


def plot_ppc(real: dict, sim: dict, label: str, outpath: str, n_ppc: int):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    real_hit = real["hit"]
    real_acc = (real["choice"][real_hit] == real["correct_side"][real_hit]).mean() \
        if real_hit.any() else float("nan")
    fig.suptitle(f"Posterior Predictive Check — {label}  (acc={real_acc:.1%})",
                 fontsize=12)

    bins = np.linspace(0, float(T_MAX), 41)
    real_rt_hit = real["rt"][real["hit"]]
    sim_rt_hit  = sim["rt"].reshape(-1)[sim["hit"].reshape(-1)]

    ax = axes[0]
    ax.hist(real_rt_hit, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="Real")
    ax.hist(sim_rt_hit, bins=bins, density=True, alpha=0.4,
            color="tomato", label=f"PPC (n={n_ppc})")
    ax.set_xlabel("RT (s)"); ax.set_ylabel("Density")
    ax.set_title("RT distribution (non-timeout)"); ax.legend(fontsize=8)

    ax = axes[1]
    for rt_arr, col, lbl in [(real_rt_hit, "steelblue", "Real"),
                             (sim_rt_hit, "tomato", "PPC")]:
        if len(rt_arr) == 0:
            continue
        s = np.sort(rt_arr)
        ax.plot(s, np.linspace(0, 1, len(s)), color=col, label=lbl, alpha=0.8)
    ax.set_xlabel("RT (s)"); ax.set_ylabel("Cumulative")
    ax.set_title("Cumulative RT"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(sim["acc_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_acc, color="steelblue", lw=2, label=f"Real ({real_acc:.2f})")
    ax.set_xlabel("Accuracy"); ax.set_title("Accuracy"); ax.legend(fontsize=8)

    ax = axes[3]
    real_p_right = real["choice"][real_hit].mean() if real_hit.any() else np.nan
    ax.hist(sim["p_right_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_p_right, color="steelblue", lw=2,
               label=f"Real ({real_p_right:.2f})")
    ax.set_xlabel("P(right)"); ax.set_title("Side preference"); ax.legend(fontsize=8)

    ax = axes[4]
    real_timeout = 1.0 - real_hit.mean()
    ax.hist(sim["timeout_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_timeout, color="steelblue", lw=2,
               label=f"Real ({real_timeout:.2f})")
    ax.set_xlabel("Timeout rate"); ax.set_title("Timeout"); ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


@dataclass
class CellResult:
    animal: str
    group: str
    stage: str
    n_sessions: int
    n_trials: int
    posterior_mean: np.ndarray
    posterior_std: np.ndarray


def fit_one_cell(
    animal: str, group: str, stage: str,
    posterior: DirectPosterior, embedding_net: torch.nn.Module,
    device: str, outdir: str,
) -> CellResult | None:
    P = max_num_pulses()
    trial_dim = 2 + P

    try:
        sessions, _meta = load_marmoset_sessions(
            csv_path=DATA_PATH, animal=animal, stage=stage,
            log_rt=bool(cfg.LOG_RT_MANUALLY), seed=SEED,
        )
    except Exception as e:
        print(f"  [skip] {animal} / {stage}: {e}")
        return None
    if not sessions:
        print(f"  [skip] {animal} / {stage}: no sessions passed filter")
        return None

    all_trials = torch.cat(
        [x.reshape(-1, trial_dim) for x in sessions], dim=0
    )
    total_n = all_trials.shape[0]
    x_combined = all_trials.reshape(1, -1).to(device, dtype=torch.float32)

    cell_dir = os.path.join(outdir, "animals", animal, stage)
    os.makedirs(cell_dir, exist_ok=True)

    with torch.no_grad():
        x_emb = embedding_net(x_combined)
    samples = posterior.sample(
        (N_POST,), x=x_emb, show_progress_bars=False,
    ).detach().cpu().numpy()
    np.save(os.path.join(cell_dir, "posterior_combined.npy"), samples)

    mean_vec = samples.mean(axis=0)
    std_vec  = samples.std(axis=0)

    with open(os.path.join(cell_dir, "summary.txt"), "w") as f:
        f.write(f"Animal: {animal}  Group: {group}  Stage: {stage}\n")
        f.write(f"N sessions: {len(sessions)}  Total trials: {total_n}\n\n")
        for d, name in enumerate(PARAM_NAMES):
            f.write(f"  {name:8s}: mean={mean_vec[d]:.4f}  std={std_vec[d]:.4f}  "
                    f"[{np.percentile(samples[:,d],5):.4f}, "
                    f"{np.percentile(samples[:,d],95):.4f}]\n")

    if stage in PPC_STAGES:
        try:
            real_combined = _extract_real(x_combined.cpu(), P)
            p_succ = STAGE_P_SUCCESS.get(stage, float(cfg.P_SUCCESS))
            sim = simulate_from_posterior(
                samples, N_PPC, T=int(cfg.NUM_TRIALS_OBS), P=P,
                p_success=p_succ, device="cpu",
            )
            plot_ppc(
                real_combined, sim,
                label=f"{animal} ({group}, stage={stage}, {total_n} trials)",
                outpath=os.path.join(cell_dir, "ppc_combined.png"),
                n_ppc=N_PPC,
            )
        except Exception:
            print(f"  [warn] PPC failed for {animal}/{stage}:")
            traceback.print_exc()

    print(f"  [ok]  {animal:14s} {stage:10s} group={group:6s} "
          f"n_trials={total_n:5d} sessions={len(sessions)} "
          f"means=[" + ", ".join(f"{v:.3f}" for v in mean_vec) + "]")

    return CellResult(
        animal=animal, group=group, stage=stage,
        n_sessions=len(sessions), n_trials=total_n,
        posterior_mean=mean_vec, posterior_std=std_vec,
    )


def per_stage_group_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in sorted(df["stage"].unique()):
        sub = df[df["stage"] == stage]
        for p in PARAM_NAMES:
            wt = sub.loc[sub["group"] == "WT", p].dropna().values
            sk = sub.loc[sub["group"] == "Shank3", p].dropna().values
            row = dict(stage=stage, param=p, n_wt=len(wt), n_shank3=len(sk))
            if len(wt) >= 2 and len(sk) >= 2:
                t_res = stats.ttest_ind(sk, wt, equal_var=False, nan_policy="omit")
                mw = stats.mannwhitneyu(sk, wt, alternative="two-sided")
                pooled_sd = math.sqrt(((len(wt) - 1) * wt.var(ddof=1)
                                       + (len(sk) - 1) * sk.var(ddof=1))
                                      / max(1, (len(wt) + len(sk) - 2)))
                row.update(
                    mean_wt=float(wt.mean()), mean_shank3=float(sk.mean()),
                    sd_wt=float(wt.std(ddof=1)), sd_shank3=float(sk.std(ddof=1)),
                    t_stat=float(t_res.statistic), t_p=float(t_res.pvalue),
                    mwu_stat=float(mw.statistic), mwu_p=float(mw.pvalue),
                    cohens_d=float((sk.mean() - wt.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
                )
            else:
                row.update(mean_wt=np.nan, mean_shank3=np.nan,
                           sd_wt=np.nan, sd_shank3=np.nan,
                           t_stat=np.nan, t_p=np.nan,
                           mwu_stat=np.nan, mwu_p=np.nan, cohens_d=np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def cross_condition_mixed_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed-effects ANOVA per parameter via statsmodels MixedLM.

    Model: y ~ C(group) * C(stage), random intercept per animal.
    Reports Wald tests for the group, stage, and group x stage terms.
    """
    import statsmodels.formula.api as smf

    rows = []
    for p in PARAM_NAMES:
        sub = df[["animal", "group", "stage", p]].dropna().rename(columns={p: "y"})
        if sub["group"].nunique() < 2 or sub["stage"].nunique() < 2:
            rows.append(dict(param=p, error="insufficient factor levels"))
            continue
        try:
            md = smf.mixedlm("y ~ C(group) * C(stage)", data=sub, groups=sub["animal"])
            mf = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            wt = mf.wald_test_terms(skip_single=False)
            tbl = wt.summary_frame()
            row = dict(param=p, n_obs=len(sub),
                       n_animals=sub["animal"].nunique(),
                       converged=bool(mf.converged),
                       loglik=float(mf.llf))
            for term, idx in [("group_p", "C(group)"),
                              ("stage_p", "C(stage)"),
                              ("interaction_p", "C(group):C(stage)")]:
                if idx in tbl.index:
                    r = tbl.loc[idx]
                    row[term]                  = float(r.get("P>chi2", r.get("pvalue", np.nan)))
                    row[term.replace("_p", "_chi2")] = float(r.get("chi2", r.get("statistic", np.nan)))
                    row[term.replace("_p", "_df")]   = float(r.get("df_constraint", r.get("df", np.nan)))
            rows.append(row)
        except Exception as e:
            rows.append(dict(param=p, error=f"{e.__class__.__name__}: {e}"))
    return pd.DataFrame(rows)


def plot_group_boxstrip(df: pd.DataFrame, tests: pd.DataFrame, outpath: str,
                        title_suffix: str = ""):
    D = len(PARAM_NAMES)
    ncols = 3
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    group_order = ["WT", "Shank3"]
    colors = {"WT": "steelblue", "Shank3": "tomato"}
    rng = np.random.default_rng(0)

    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        data = [df.loc[df["group"] == g, name].dropna().values for g in group_order]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.5))
        for patch, g in zip(bp["boxes"], group_order):
            patch.set_facecolor(colors[g]); patch.set_alpha(0.35)
        for i, g in enumerate(group_order):
            y = data[i]
            x = i + (rng.random(len(y)) - 0.5) * 0.22
            ax.scatter(x, y, s=30, color=colors[g], edgecolors="black",
                       linewidths=0.5, alpha=0.9, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(group_order)
        ax.set_title(name)
        row = tests.loc[tests["param"] == name]
        if len(row):
            r = row.iloc[0]
            ax.text(0.02, 0.98,
                    f"t p={r['t_p']:.3g}\nMWU p={r['mwu_p']:.3g}\nd={r['cohens_d']:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7,
                              edgecolor="lightgray"))
    for ax in axes[D:]:
        ax.set_visible(False)
    n_wt = (df["group"] == "WT").sum()
    n_sk = (df["group"] == "Shank3").sum()
    fig.suptitle(f"Posterior means by group (n WT={n_wt}, Shank3={n_sk}) {title_suffix}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pairgrid(df: pd.DataFrame, outpath: str):
    D = len(PARAM_NAMES)
    fig, axes = plt.subplots(D, D, figsize=(2.3 * D, 2.3 * D))
    colors = {"WT": "steelblue", "Shank3": "tomato"}
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                for g, c in colors.items():
                    vals = df.loc[df["group"] == g, PARAM_NAMES[i]].dropna().values
                    if len(vals):
                        ax.hist(vals, bins=12, color=c, alpha=0.5, label=g)
                if i == 0:
                    ax.legend(fontsize=7)
            else:
                for g, c in colors.items():
                    sub = df.loc[df["group"] == g]
                    ax.scatter(sub[PARAM_NAMES[j]], sub[PARAM_NAMES[i]],
                               s=18, color=c, edgecolors="black",
                               linewidths=0.3, alpha=0.8)
            if i == D - 1:
                ax.set_xlabel(PARAM_NAMES[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(PARAM_NAMES[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Per-animal posterior means (scatter matrix by group)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_animal_bars(df: pd.DataFrame, outpath: str):
    D = len(PARAM_NAMES)
    ncols = 2
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.8 * nrows))
    axes = np.array(axes).flatten()
    colors = {"WT": "steelblue", "Shank3": "tomato"}
    df_sorted = df.sort_values(["group", "animal"]).reset_index(drop=True)
    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        y = df_sorted[name].values
        x = np.arange(len(df_sorted))
        bar_colors = [colors[g] for g in df_sorted["group"]]
        ax.bar(x, y, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted["animal"], rotation=90, fontsize=6)
        ax.set_title(name)
        ax.set_ylabel("posterior mean")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=g) for g, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.95, 1.0))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_x_group_lines(df: pd.DataFrame, stages_order: list[str],
                             tests_xc: pd.DataFrame, outpath: str):
    """Mean +/- SEM trajectory across stages for each group, per parameter."""
    D = len(PARAM_NAMES)
    ncols = 3
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    colors = {"WT": "steelblue", "Shank3": "tomato"}
    x_pos = {s: i for i, s in enumerate(stages_order)}

    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        for g, c in colors.items():
            xs, ys, sems = [], [], []
            for s in stages_order:
                vals = df.loc[(df["group"] == g) & (df["stage"] == s), name].dropna().values
                if len(vals) == 0:
                    continue
                xs.append(x_pos[s])
                ys.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            xs, ys, sems = np.array(xs), np.array(ys), np.array(sems)
            ax.errorbar(xs, ys, yerr=sems, marker="o", color=c, label=g,
                        capsize=3, lw=1.5)
        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(stages_order, rotation=30, fontsize=8)
        ax.set_title(name)
        row = tests_xc.loc[tests_xc["param"] == name]
        if len(row) and "group_p" in row.columns:
            r = row.iloc[0]
            txt = (f"group p={r.get('group_p', np.nan):.3g}\n"
                   f"stage p={r.get('stage_p', np.nan):.3g}\n"
                   f"g×s p={r.get('interaction_p', np.nan):.3g}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75,
                              edgecolor="lightgray"))
        if d == 0:
            ax.legend(fontsize=8)
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle("Posterior mean trajectories across stages (mean ± SEM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_x_group_violins(df: pd.DataFrame, stages_order: list[str], outpath: str):
    D = len(PARAM_NAMES)
    ncols = 3
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    colors = {"WT": "steelblue", "Shank3": "tomato"}

    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        positions, data_list, face_colors = [], [], []
        for i, s in enumerate(stages_order):
            for k, g in enumerate(["WT", "Shank3"]):
                vals = df.loc[(df["stage"] == s) & (df["group"] == g), name].dropna().values
                if len(vals) >= 2:
                    positions.append(i + (k - 0.5) * 0.4)
                    data_list.append(vals)
                    face_colors.append(colors[g])
        if data_list:
            parts = ax.violinplot(data_list, positions=positions, widths=0.35,
                                  showmeans=True, showmedians=False)
            for pc, fc in zip(parts["bodies"], face_colors):
                pc.set_facecolor(fc); pc.set_alpha(0.5); pc.set_edgecolor("black")
        ax.set_xticks(range(len(stages_order)))
        ax.set_xticklabels(stages_order, rotation=30, fontsize=8)
        ax.set_title(name)
    for ax in axes[D:]:
        ax.set_visible(False)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=g, alpha=0.6) for g, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("Posterior mean distributions per stage × group", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.97, 1.0))
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    os.makedirs(OUTDIR, exist_ok=True)

    df_all = pd.read_csv(
        DATA_PATH, compression="infer",
        dtype={"flashes_left": str, "flashes_right": str},
    )

    df_filt = df_all[df_all["stage"].isin(STAGES)]
    if ANIMALS_FILTER:
        wanted = {a.strip() for a in ANIMALS_FILTER.split(",") if a.strip()}
        df_filt = df_filt[df_filt["name"].isin(wanted)]

    roster = (df_filt[["name", "group", "stage"]]
              .drop_duplicates()
              .sort_values(["group", "name", "stage"])
              .reset_index(drop=True))
    if len(roster) == 0:
        raise RuntimeError(f"No (animal,stage) cells found for stages={STAGES}")
    print(f"Found {len(roster)} (animal, stage) cells across "
          f"{roster['name'].nunique()} animals and {roster['stage'].nunique()} stages.")

    prior_theta = build_prior_theta_lapse()
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f"\nLoading NPE from {model_path} on {device} ...")
    de, embedding_net, _saved_cfg, _T = load_npe_decoupled(
        model_path, prior_theta=prior_theta, device=device,
    )
    posterior = DirectPosterior(
        posterior_estimator=de, prior=prior_theta, device=device,
    )

    results: list[CellResult] = []
    for _, r in roster.iterrows():
        animal, group, stage = r["name"], r["group"], r["stage"]
        try:
            cr = fit_one_cell(
                animal=animal, group=group, stage=stage,
                posterior=posterior, embedding_net=embedding_net,
                device=device, outdir=OUTDIR,
            )
        except Exception:
            print(f"  [err] {animal} / {stage} failed:")
            traceback.print_exc()
            continue
        if cr is not None:
            results.append(cr)

    if not results:
        raise RuntimeError("No cells produced posterior samples.")

    rows = []
    for cr in results:
        d = dict(animal=cr.animal, group=cr.group, stage=cr.stage,
                 n_sessions=cr.n_sessions, n_trials=cr.n_trials)
        for i, name in enumerate(PARAM_NAMES):
            d[name]         = float(cr.posterior_mean[i])
            d[f"{name}_sd"] = float(cr.posterior_std[i])
        rows.append(d)
    df_means = pd.DataFrame(rows)
    means_csv = os.path.join(OUTDIR, "all_animals_all_stages.csv")
    df_means.to_csv(means_csv, index=False)
    print(f"\nSaved per-(animal,stage) means: {means_csv}")

    tests_per_stage = per_stage_group_tests(df_means)
    tests_per_stage_csv = os.path.join(OUTDIR, "per_stage_group_tests.csv")
    tests_per_stage.to_csv(tests_per_stage_csv, index=False)
    print(f"Saved per-stage group tests: {tests_per_stage_csv}")

    tests_xc = cross_condition_mixed_anova(df_means)
    tests_xc_csv = os.path.join(OUTDIR, "cross_condition_tests.csv")
    tests_xc.to_csv(tests_xc_csv, index=False)
    print(f"Saved cross-condition mixed ANOVA: {tests_xc_csv}")

    print("\n=== Cross-condition mixed ANOVA (group × stage on each parameter) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(tests_xc.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    for stage in sorted(df_means["stage"].unique()):
        sub = df_means[df_means["stage"] == stage]
        if len(sub) < 4:
            continue
        sd = os.path.join(OUTDIR, "per_stage", stage)
        os.makedirs(sd, exist_ok=True)
        ts = (tests_per_stage[tests_per_stage["stage"] == stage]
              .reset_index(drop=True))
        plot_group_boxstrip(sub, ts,
                            outpath=os.path.join(sd, "group_box_strip.png"),
                            title_suffix=f"— stage {stage}")
        plot_pairgrid(sub, outpath=os.path.join(sd, "group_pairgrid.png"))
        plot_per_animal_bars(sub, outpath=os.path.join(sd, "per_animal_bars.png"))

    plot_stage_x_group_lines(df_means, STAGES, tests_xc,
                             outpath=os.path.join(OUTDIR, "stage_x_group_lines.png"))
    plot_stage_x_group_violins(df_means, STAGES,
                               outpath=os.path.join(OUTDIR, "stage_x_group_violins.png"))

    manifest = dict(
        stages=STAGES, ppc_stages=PPC_STAGES,
        n_post=N_POST, n_ppc=N_PPC, seed=SEED,
        model_dir=MODEL_DIR, model_file=MODEL_FILE,
        data_path=DATA_PATH,
        n_cells_fit=len(results),
        n_animals=int(df_means["animal"].nunique()),
        per_stage_n=df_means.groupby("stage")["animal"].nunique().to_dict(),
        per_group_n=df_means.groupby("group")["animal"].nunique().to_dict(),
    )
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
