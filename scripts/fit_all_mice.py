"""Fit a trained mouse NPE to every mouse at every stage, run posterior
predictive checks, and test parameters across genotype (WT / SHANK3-HET /
SHANK3-HOM) and condition (stage_cat).

Mirrors scripts/fit_all_marmosets_all_stages.py but for the mouse task:
  * data come from ALLspecies_trials_combined.csv (specie == "Mouse")
  * stages are stage_cat strings: 100-0 (p=1.0), 90-10 (p=0.9), 80-20 (p=0.8)
  * three genotype groups instead of two
  * the mouse timescale (P=50, T_MAX=5 s) is set by SPECIES=mouse, which is
    asserted against the loaded checkpoint to prevent a timescale mismatch.

Usage
-----
    SPECIES=mouse MODEL_NAME=lapse_noleak_ar_mouse python scripts/fit_all_mice.py
    SPECIES=mouse MODEL_NAME=lapse_noleak_mouse  OUTDIR=group_outputs_mice_noar \
        python scripts/fit_all_mice.py

Outputs (default OUTDIR=group_outputs_mice)
-------------------------------------------
  manifest.json
  all_mice_all_stages.csv            # tidy: one row per (animal, stage)
  per_stage_group_tests.csv          # pairwise Welch + MWU per (param, stage, group-pair)
  cross_condition_tests.csv          # genotype × stage cluster-robust ANOVA per param
  stage_x_group_lines.png            # mean ± SEM trajectory across stages
  per_stage/<stage>/group_box_strip.png
  animals/<animal>/<stage>/posterior_combined.npy, summary.txt, ppc_combined.png
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

from sbi_for_diffusion_models.model_specs import select_model
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses, mask_unperceived_pulses
from sbi_for_diffusion_models.ppc import plot_ppc_panels
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions,
    simulate_training_sessions_ar,
    trial_feature_dim,
)
from sbi_for_diffusion_models.mnpe import load_npe_decoupled
from sbi_for_diffusion_models.load_mouse import load_mouse_sessions, STAGE_CAT_P_SUCCESS
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL, SPECIES as CFG_SPECIES

cfg = RUN_CONFIG_PARAMS

ALL_STAGES = ["100-0", "90-10", "80-20"]
GROUP_ORDER = ["WT", "SHANK3-HET", "SHANK3-HOM"]
GROUP_COLORS = {"WT": "steelblue", "SHANK3-HET": "mediumseagreen", "SHANK3-HOM": "tomato"}

STAGES_ENV = os.environ.get("STAGES", ",".join(ALL_STAGES))
STAGES     = [s.strip() for s in STAGES_ENV.split(",") if s.strip()]
PPC_STAGES = [s.strip() for s in os.environ.get("PPC_STAGES", "80-20").split(",") if s.strip()]
N_POST     = int(os.environ.get("N_POST", "5000"))
N_PPC      = int(os.environ.get("N_PPC",  "200"))
SEED       = int(os.environ.get("SEED",   "0"))
OUTDIR     = os.environ.get("OUTDIR",     "group_outputs_mice")
MODEL_DIR  = os.path.expanduser(os.environ.get("MODEL_DIR", "models"))
MODEL_NAME = os.environ.get("MODEL_NAME", "lapse_noleak_ar_mouse")
DATA_PATH  = os.environ.get(
    "DATA_PATH",
    "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/ALLspecies_trials_combined.csv",
)
ANIMALS_FILTER = os.environ.get("ANIMALS", "").strip()


def _default_model_file(model_name: str, cfg) -> str:
    values = "_".join(f"{int(round(v * 100)):03d}" for v in cfg.P_SUCCESS_TRAIN_VALUES)
    return f"npe_{model_name}_mixed_{values}.pt"


MODEL_FILE = os.environ.get("MODEL_FILE", _default_model_file(MODEL_NAME, cfg))

_, _, PARAM_NAMES_TUPLE, _MODEL_TAG, AUTOREGRESSIVE = select_model(MODEL_NAME)
PARAM_NAMES = list(PARAM_NAMES_TUPLE)


# ---------------------------------------------------------------------------
# PPC
# ---------------------------------------------------------------------------

def _random_pulses(T: int, P: int, p_success: float, dev: torch.device):
    """Exclusive ±1 pulses (mouse/marmoset scheme): one side flashes per bin."""
    correct_side = (torch.randint(0, 2, (T,), device=dev) * 2 - 1).float()
    is_correct = torch.rand(T, P, device=dev) < p_success
    pulses = torch.where(
        is_correct,
        correct_side.unsqueeze(1).expand(T, P),
        -correct_side.unsqueeze(1).expand(T, P),
    )
    return pulses, correct_side


@torch.no_grad()
def simulate_from_posterior(posterior_samples: np.ndarray, n_ppc: int,
                            T: int, P: int, p_success: float,
                            *, simulate_batch_fn, autoregressive: bool,
                            device: str = "cpu") -> dict:
    dev = torch.device(device)
    replace = len(posterior_samples) < n_ppc
    idx = np.random.choice(len(posterior_samples), size=n_ppc, replace=replace)
    theta = torch.tensor(posterior_samples[idx], dtype=torch.float32, device=dev)

    log_tmax = math.log(T_MAX) if cfg.LOG_RT_MANUALLY else T_MAX
    trial_dim_ar = trial_feature_dim(P, ar=True)

    all_rt, all_choice, all_hit, all_correct, all_net = [], [], [], [], []
    all_prev_choice, all_prev_outcome = [], []

    if autoregressive:
        _, x_flat = simulate_training_sessions_ar(
            prior_theta=None, num_sessions=n_ppc, num_trials=T,
            simulate_batch_fn=simulate_batch_fn, device=dev,
            mu_sensory=float(cfg.MU_SENSORY), p_success=float(p_success),
            P=P, log_rt=bool(cfg.LOG_RT_MANUALLY),
            seed=int(np.random.randint(0, 2**31 - 1)),
            theta=theta, warn_on_timeouts=False,
        )
        x_3d = x_flat.view(n_ppc, T, trial_dim_ar).cpu().numpy()
        for i in range(n_ppc):
            log_rt_i = x_3d[i, :, 0]
            choice_i = x_3d[i, :, 1].astype(np.int32)
            pulses_i = x_3d[i, :, 4:]
            rt_i = np.exp(log_rt_i) if cfg.LOG_RT_MANUALLY else log_rt_i
            hit_i = log_rt_i < log_tmax - 1e-4
            all_rt.append(rt_i.astype(np.float32)); all_choice.append(choice_i)
            all_hit.append(hit_i)
            all_correct.append((pulses_i.sum(axis=-1) > 0).astype(np.int32))
            all_net.append(pulses_i.sum(axis=-1).astype(np.float32))
            all_prev_choice.append(x_3d[i, :, 2].astype(np.float32))
            all_prev_outcome.append(x_3d[i, :, 3].astype(np.float32))
    else:
        for i in range(n_ppc):
            th_i = theta[i].unsqueeze(0).expand(T, -1)
            pulses, correct_side_i = _random_pulses(T, P, p_success, dev)
            x_raw_i, hit_i, _ = simulate_batch_fn(
                th_i, pulse_sides=pulses, mu_sensory=float(cfg.MU_SENSORY),
                p_success=p_success,
            )
            rt_i = x_raw_i[:, 0]
            # mask unperceived pulses (after RT) so net evidence matches the
            # response-terminated real data
            pulses_m = mask_unperceived_pulses(pulses, rt_i, float(PULSE_INTERVAL))
            pulses_m = torch.nan_to_num(pulses_m, nan=0.0).cpu().numpy()
            all_rt.append(rt_i.cpu().numpy().astype(np.float32))
            all_choice.append(x_raw_i[:, 1].cpu().numpy().astype(np.int32))
            all_hit.append(hit_i.cpu().numpy().astype(bool))
            all_correct.append(((correct_side_i > 0).long()).cpu().numpy().astype(np.int32))
            all_net.append(pulses_m.sum(axis=-1).astype(np.float32))

    rt = np.stack(all_rt); choice = np.stack(all_choice)
    hit = np.stack(all_hit); correct_side = np.stack(all_correct)
    net_evidence = np.stack(all_net)

    acc_per_draw = np.array([
        (choice[i][hit[i]] == correct_side[i][hit[i]]).mean() if hit[i].any() else np.nan
        for i in range(n_ppc)
    ])
    p_right_per_draw = np.array([
        choice[i][hit[i]].mean() if hit[i].any() else np.nan for i in range(n_ppc)
    ])
    out = {"rt": rt, "choice": choice, "hit": hit, "correct_side": correct_side,
           "net_evidence": net_evidence, "acc_per_draw": acc_per_draw,
           "p_right_per_draw": p_right_per_draw,
           "timeout_per_draw": 1.0 - hit.mean(axis=1)}
    if autoregressive:
        out["choice_signed"] = (2 * choice - 1).astype(np.float32)
        out["prev_choice"] = np.stack(all_prev_choice)
        out["prev_outcome"] = np.stack(all_prev_outcome)
    return out


def _extract_real(x_flat: torch.Tensor, P: int, autoregressive: bool = False) -> dict:
    trial_dim = trial_feature_dim(P, ar=autoregressive)
    T = x_flat.shape[1] // trial_dim
    x = x_flat.reshape(T, trial_dim).numpy()
    log_rt = x[:, 0]; choice = x[:, 1].astype(int)
    pulses = x[:, (4 if autoregressive else 2):]
    rt = np.exp(log_rt) if cfg.LOG_RT_MANUALLY else log_rt
    hit = rt < float(T_MAX) - 0.01
    out = {"rt": rt, "choice": choice, "hit": hit,
           "correct_side": (pulses.sum(axis=1) > 0).astype(int),
           "net_evidence": pulses.sum(axis=1).astype(np.float32),
           "n_trials": T}
    if autoregressive:
        out["choice_signed"] = (2 * choice - 1).astype(np.float32)
        out["prev_choice"] = x[:, 2].astype(np.float32)
        out["prev_outcome"] = x[:, 3].astype(np.float32)
    return out


def plot_ppc(real: dict, sim: dict, label: str, outpath: str, n_ppc: int,
             autoregressive: bool = False):
    return plot_ppc_panels(real, sim, label=label, outpath=outpath,
                           t_max=float(T_MAX), ar=autoregressive, n_ppc=n_ppc)


# ---------------------------------------------------------------------------
# Per-cell fit
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    animal: str
    group: str
    stage: str
    n_sessions: int
    n_trials: int
    posterior_mean: np.ndarray
    posterior_std: np.ndarray


def fit_one_cell(animal: str, group: str, stage: str,
                 posterior: DirectPosterior, embedding_net: torch.nn.Module,
                 device: str, outdir: str, *, simulate_batch_fn, autoregressive: bool) -> CellResult | None:
    P = max_num_pulses()
    trial_dim = trial_feature_dim(P, ar=autoregressive)

    try:
        sessions, _meta = load_mouse_sessions(
            csv_path=DATA_PATH, animal=animal, stage=stage,
            log_rt=bool(cfg.LOG_RT_MANUALLY), seed=SEED, autoregressive=autoregressive,
        )
    except Exception as e:
        print(f"  [skip] {animal} / {stage}: {e}")
        return None
    if not sessions:
        print(f"  [skip] {animal} / {stage}: no sessions passed filter")
        return None

    all_trials = torch.cat([x.reshape(-1, trial_dim) for x in sessions], dim=0)
    total_n = all_trials.shape[0]
    x_combined = all_trials.reshape(1, -1).to(device, dtype=torch.float32)

    cell_dir = os.path.join(outdir, "animals", animal, stage)
    os.makedirs(cell_dir, exist_ok=True)

    with torch.no_grad():
        x_emb = embedding_net(x_combined)
    samples = posterior.sample((N_POST,), x=x_emb, show_progress_bars=False).detach().cpu().numpy()
    np.save(os.path.join(cell_dir, "posterior_combined.npy"), samples)

    mean_vec = samples.mean(axis=0); std_vec = samples.std(axis=0)
    with open(os.path.join(cell_dir, "summary.txt"), "w") as f:
        f.write(f"Animal: {animal}  Group: {group}  Stage: {stage}\n")
        f.write(f"N sessions: {len(sessions)}  Total trials: {total_n}\n\n")
        for d, name in enumerate(PARAM_NAMES):
            f.write(f"  {name:8s}: mean={mean_vec[d]:.4f}  std={std_vec[d]:.4f}  "
                    f"[{np.percentile(samples[:,d],5):.4f}, {np.percentile(samples[:,d],95):.4f}]\n")

    if stage in PPC_STAGES:
        try:
            real_combined = _extract_real(x_combined.cpu(), P, autoregressive=autoregressive)
            p_succ = STAGE_CAT_P_SUCCESS.get(stage, float(cfg.P_SUCCESS))
            sim = simulate_from_posterior(
                samples, N_PPC, T=int(cfg.NUM_TRIALS_OBS), P=P, p_success=p_succ,
                device="cpu", simulate_batch_fn=simulate_batch_fn, autoregressive=autoregressive,
            )
            plot_ppc(real_combined, sim,
                     label=f"{animal} ({group}, stage={stage}, {total_n} trials)",
                     outpath=os.path.join(cell_dir, "ppc_combined.png"), n_ppc=N_PPC,
                     autoregressive=autoregressive)
        except Exception:
            print(f"  [warn] PPC failed for {animal}/{stage}:")
            traceback.print_exc()

    print(f"  [ok]  {animal:14s} {stage:8s} group={group:11s} "
          f"n_trials={total_n:6d} sessions={len(sessions)} "
          f"means=[" + ", ".join(f"{v:.3f}" for v in mean_vec) + "]")
    return CellResult(animal=animal, group=group, stage=stage,
                      n_sessions=len(sessions), n_trials=total_n,
                      posterior_mean=mean_vec, posterior_std=std_vec)


# ---------------------------------------------------------------------------
# Group analyses
# ---------------------------------------------------------------------------

def per_stage_group_tests(df: pd.DataFrame, groups=GROUP_ORDER) -> pd.DataFrame:
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
    rows = []
    for stage in sorted(df["stage"].unique()):
        sub = df[df["stage"] == stage]
        for a, b in pairs:
            for p in PARAM_NAMES:
                xa = sub.loc[sub["group"] == a, p].dropna().values
                xb = sub.loc[sub["group"] == b, p].dropna().values
                row = dict(stage=stage, group_a=a, group_b=b, param=p,
                           n_a=len(xa), n_b=len(xb))
                if len(xa) >= 2 and len(xb) >= 2:
                    t_res = stats.ttest_ind(xb, xa, equal_var=False, nan_policy="omit")
                    mw = stats.mannwhitneyu(xb, xa, alternative="two-sided")
                    pooled_sd = math.sqrt(((len(xa) - 1) * xa.var(ddof=1)
                                           + (len(xb) - 1) * xb.var(ddof=1))
                                          / max(1, (len(xa) + len(xb) - 2)))
                    row.update(mean_a=float(xa.mean()), mean_b=float(xb.mean()),
                               t_stat=float(t_res.statistic), t_p=float(t_res.pvalue),
                               mwu_stat=float(mw.statistic), mwu_p=float(mw.pvalue),
                               cohens_d=float((xb.mean() - xa.mean()) / pooled_sd) if pooled_sd > 0 else np.nan)
                else:
                    row.update(mean_a=np.nan, mean_b=np.nan, t_stat=np.nan, t_p=np.nan,
                               mwu_stat=np.nan, mwu_p=np.nan, cohens_d=np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def cross_condition_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Genotype × stage cluster-robust (clustered on animal) ANOVA per parameter."""
    import statsmodels.formula.api as smf
    rows = []
    for p in PARAM_NAMES:
        sub = df[["animal", "group", "stage", p]].dropna().rename(columns={p: "y"})
        if sub["group"].nunique() < 2 or sub["stage"].nunique() < 2:
            rows.append(dict(param=p, error="insufficient factor levels")); continue
        try:
            mf = smf.ols("y ~ C(group) * C(stage)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["animal"]})
            tbl = mf.wald_test_terms(skip_single=False).summary_frame()
            row = dict(param=p, n_obs=len(sub), n_animals=sub["animal"].nunique(),
                       method="ols+cluster_robust", rsq=float(mf.rsquared))
            for term, idx in [("group_p", "C(group)"), ("stage_p", "C(stage)"),
                              ("interaction_p", "C(group):C(stage)")]:
                if idx in tbl.index:
                    r = tbl.loc[idx]
                    row[term] = float(r.get("P>chi2", r.get("pvalue", np.nan)))
                    row[term.replace("_p", "_chi2")] = float(r.get("chi2", r.get("statistic", np.nan)))
            rows.append(row)
        except Exception as e:
            rows.append(dict(param=p, error=f"{e.__class__.__name__}: {e}"))
    return pd.DataFrame(rows)


def plot_group_boxstrip(df: pd.DataFrame, outpath: str, title_suffix: str = ""):
    D = len(PARAM_NAMES); ncols = 3; nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten(); rng = np.random.default_rng(0)
    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        data = [df.loc[df["group"] == g, name].dropna().values for g in GROUP_ORDER]
        bp = ax.boxplot(data, positions=list(range(len(GROUP_ORDER))), widths=0.55,
                        patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.5))
        for patch, g in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.35)
        for i, g in enumerate(GROUP_ORDER):
            y = data[i]; x = i + (rng.random(len(y)) - 0.5) * 0.22
            ax.scatter(x, y, s=22, color=GROUP_COLORS[g], edgecolors="black",
                       linewidths=0.4, alpha=0.85, zorder=3)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER, rotation=20, fontsize=8); ax.set_title(name)
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle(f"Posterior means by genotype {title_suffix}", fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_stage_x_group_lines(df: pd.DataFrame, stages_order, tests_xc, outpath: str):
    D = len(PARAM_NAMES); ncols = 3; nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    x_pos = {s: i for i, s in enumerate(stages_order)}
    for d, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        for g in GROUP_ORDER:
            xs, ys, sems = [], [], []
            for s in stages_order:
                vals = df.loc[(df["group"] == g) & (df["stage"] == s), name].dropna().values
                if len(vals) == 0:
                    continue
                xs.append(x_pos[s]); ys.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=sems, marker="o", color=GROUP_COLORS[g], label=g, capsize=3, lw=1.5)
        ax.set_xticks(list(x_pos.values())); ax.set_xticklabels(stages_order, rotation=20, fontsize=8)
        ax.set_title(name)
        row = tests_xc.loc[tests_xc["param"] == name]
        if len(row) and "group_p" in row.columns:
            r = row.iloc[0]
            ax.text(0.02, 0.98, f"group p={r.get('group_p', np.nan):.3g}\n"
                    f"stage p={r.get('stage_p', np.nan):.3g}\n"
                    f"g×s p={r.get('interaction_p', np.nan):.3g}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="lightgray"))
        if d == 0:
            ax.legend(fontsize=8)
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle("Posterior mean trajectories across stages (mean ± SEM)", fontsize=12)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if CFG_SPECIES != "mouse":
        raise RuntimeError(
            f"SPECIES={CFG_SPECIES!r}; run with SPECIES=mouse so the timescale is "
            f"P=50 / T_MAX=5 s. Set SPECIES=mouse in the environment."
        )
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    os.makedirs(OUTDIR, exist_ok=True)

    df_all = pd.read_csv(DATA_PATH, compression="infer",
                         dtype={"flashes_left": str, "flashes_right": str}, low_memory=False)
    df_all = df_all[df_all["specie"] == "Mouse"]
    df_filt = df_all[df_all["stage_cat"].isin(STAGES)]
    if ANIMALS_FILTER:
        wanted = {a.strip() for a in ANIMALS_FILTER.split(",") if a.strip()}
        df_filt = df_filt[df_filt["name"].isin(wanted)]

    roster = (df_filt[["name", "group", "stage_cat"]].drop_duplicates()
              .rename(columns={"stage_cat": "stage"})
              .sort_values(["group", "name", "stage"]).reset_index(drop=True))
    if len(roster) == 0:
        raise RuntimeError(f"No (animal,stage) cells for stages={STAGES}")
    print(f"Found {len(roster)} (animal, stage) cells across "
          f"{roster['name'].nunique()} mice and {roster['stage'].nunique()} stages.")
    print(f"Genotypes: {dict(roster.groupby('group')['name'].nunique())}")

    simulate_batch_fn, prior_theta, _pn, _tag, autoregressive = select_model(MODEL_NAME)
    if hasattr(prior_theta, "to"):
        prior_theta.to(dev)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f"\nLoading NPE from {model_path} (MODEL_NAME={MODEL_NAME}, ar={autoregressive}) ...")
    de, embedding_net, saved_cfg, _T = load_npe_decoupled(model_path, prior_theta=prior_theta, device=device)

    saved_ar = bool(getattr(saved_cfg, "AUTOREGRESSIVE", False))
    if saved_ar != autoregressive:
        raise RuntimeError(f"AR mismatch: MODEL_NAME implies {autoregressive}, checkpoint {saved_ar}.")
    saved_species = getattr(saved_cfg, "SPECIES", None)
    if saved_species is not None and saved_species != "mouse":
        raise RuntimeError(f"Checkpoint SPECIES={saved_species!r} != 'mouse'; timescale would mismatch.")
    posterior = DirectPosterior(posterior_estimator=de, prior=prior_theta, device=device)

    results: list[CellResult] = []
    for _, r in roster.iterrows():
        try:
            cr = fit_one_cell(r["name"], r["group"], r["stage"], posterior, embedding_net,
                              device, OUTDIR, simulate_batch_fn=simulate_batch_fn,
                              autoregressive=autoregressive)
        except Exception:
            print(f"  [err] {r['name']} / {r['stage']} failed:")
            traceback.print_exc(); continue
        if cr is not None:
            results.append(cr)

    if not results:
        raise RuntimeError("No cells produced posterior samples.")

    rows = []
    for cr in results:
        d = dict(animal=cr.animal, group=cr.group, stage=cr.stage,
                 n_sessions=cr.n_sessions, n_trials=cr.n_trials)
        for i, name in enumerate(PARAM_NAMES):
            d[name] = float(cr.posterior_mean[i]); d[f"{name}_sd"] = float(cr.posterior_std[i])
        rows.append(d)
    df_means = pd.DataFrame(rows)
    means_csv = os.path.join(OUTDIR, "all_mice_all_stages.csv")
    df_means.to_csv(means_csv, index=False)
    print(f"\nSaved per-(animal,stage) means: {means_csv}")

    tests_per_stage = per_stage_group_tests(df_means)
    tests_per_stage.to_csv(os.path.join(OUTDIR, "per_stage_group_tests.csv"), index=False)

    tests_xc = cross_condition_anova(df_means)
    tests_xc.to_csv(os.path.join(OUTDIR, "cross_condition_tests.csv"), index=False)
    print("\n=== Genotype × stage ANOVA (cluster-robust on animal) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(tests_xc.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    for stage in sorted(df_means["stage"].unique()):
        sub = df_means[df_means["stage"] == stage]
        if len(sub) < 4:
            continue
        sd = os.path.join(OUTDIR, "per_stage", stage); os.makedirs(sd, exist_ok=True)
        plot_group_boxstrip(sub, os.path.join(sd, "group_box_strip.png"),
                            title_suffix=f"— stage {stage}")

    plot_stage_x_group_lines(df_means, STAGES, tests_xc,
                             os.path.join(OUTDIR, "stage_x_group_lines.png"))

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(dict(species="mouse", model_name=MODEL_NAME, model_file=MODEL_FILE,
                       stages=STAGES, ppc_stages=PPC_STAGES, n_post=N_POST, n_ppc=N_PPC,
                       seed=SEED, param_names=PARAM_NAMES, n_cells_fit=len(results),
                       n_animals=int(df_means["animal"].nunique()),
                       per_group_n=df_means.groupby("group")["animal"].nunique().to_dict(),
                       per_stage_n=df_means.groupby("stage")["animal"].nunique().to_dict()), f, indent=2)
    print(f"\nAll done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
