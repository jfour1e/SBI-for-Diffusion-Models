"""Compare and select among trained mouse NPE models per animal × stage.

For each (animal, stage) cell and each model, this script:
  * fits the model's posterior to the combined-session data,
  * runs a posterior-predictive check and summarises it to a scalar discrepancy,
  * estimates the expected log predictive density. Default is honest K-fold
    held-out cross-validation (ELPD_METHOD=cv): the amortized posterior is
    re-conditioned on the train fold and the held-out fold is scored by its
    posterior-predictive density — no importance sampling, no Pareto-k, no
    log-lik flooring. ELPD_METHOD=psis falls back to the legacy synthetic-
    likelihood PSIS-LOO (kept for comparison; its Pareto-k is masked by the
    floor — see sbi_for_diffusion_models.model_comparison).

It then selects, per cell, the model with the highest elpd_loo and reports the
elpd difference (with SE) to the runner-up, and aggregates wins by genotype.

Default models compared: lapse_noleak_ar_mouse vs lapse_noleak_mouse — i.e.
"does the autoregressive (history) term improve predictive fit?".

Usage
-----
    SPECIES=mouse python scripts/model_comparison.py
    SPECIES=mouse MODELS=lapse_noleak_ar_mouse,lapse_noleak_mouse \
        ELPD_DRAWS=100 ELPD_REPLICATES=300 ELPD_MAX_TRIALS=1000 \
        python scripts/model_comparison.py

Key env vars
------------
    MODELS            comma list of model names (default the two mouse models)
    STAGES            comma list of stage_cat (default 100-0,90-10,80-20)
    ANIMALS           optional comma list to restrict animals
    OUTDIR            output dir (default model_comparison_mice)
    MODEL_DIR         dir holding npe_<model>_mixed_*.pt checkpoints
    DO_ELPD / DO_PPC  toggle each analysis (default 1)
    ELPD_METHOD       "cv" (held-out CV, default) or "psis" (legacy PSIS-LOO)
    ELPD_FOLDS        K for cv (default 5)
    ELPD_RT_DENSITY   "lognormal" (default) or "kde" for the RT term
    ELPD_DRAWS        posterior draws per fit/fold (default 100)
    ELPD_REPLICATES   MC replicates per draw per trial (default 300)
    ELPD_MAX_TRIALS   subsample observed trials for ELPD (default 1000)
    N_PPC             posterior-predictive sessions (default 200)
    N_POST            posterior samples drawn per fit (default 2000)
"""
from __future__ import annotations

import os
import json
import traceback

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.inference.posteriors import DirectPosterior

from sbi_for_diffusion_models.model_specs import select_model
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.data_simulator import (
    simulate_training_sessions, simulate_training_sessions_ar, trial_feature_dim,
)
from sbi_for_diffusion_models.mnpe import load_npe_decoupled
from sbi_for_diffusion_models.load_mouse import load_mouse_sessions, STAGE_CAT_P_SUCCESS
from sbi_for_diffusion_models.load_marmoset import load_marmoset_sessions
from sbi_for_diffusion_models.load_human import load_human_sessions_compare
from sbi_for_diffusion_models.model_comparison import (
    extract_observed, synthetic_loglik_matrix, elpd_psis_loo, crossval_elpd,
    ppc_discrepancy, _session_summary,
)
from sbi_for_diffusion_models.ppc import (
    rt_distribution_distance, ar_choice_stats, ar_stats_from_dict,
)
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, T_MAX, SPECIES as CFG_SPECIES

cfg = RUN_CONFIG_PARAMS

_ALLSPECIES_CSV = "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/ALLspecies_trials_combined.csv"
_MARMOSET_CSV = "/projectnb/ssmsvi/rsenne/data_marmoset/marmoset_data.csv.gz"

# Species adapter: which loader, stage column, p_success map, and sensible
# defaults. ELPD itself is species-agnostic (it re-uses observed pulses and, for
# AR, observed history); only the data source and PPC p_success differ.
_SPECIES_ADAPTERS = {
    "mouse": dict(
        load_fn=load_mouse_sessions, stage_p=STAGE_CAT_P_SUCCESS,
        stage_col="stage_cat", specie_filter="Mouse",
        default_stages="100-0,90-10,80-20",
        default_models="lapse_noleak_ar_mouse,lapse_noleak_mouse",
        default_data=_ALLSPECIES_CSV, default_outdir="model_comparison_mice"),
    "marmoset": dict(
        load_fn=load_marmoset_sessions,
        stage_p={"100-0": 1.0, "90-10": 0.9, "80-20": 0.8, "70-30": 0.7,
                 "60-40": 0.6, "randomProb": 0.7},
        stage_col="stage", specie_filter=None,
        default_stages="100-0,90-10,80-20,70-30,60-40",
        default_models="lapse_noleak_ar,lapse_noleak,lapse",
        default_data=_MARMOSET_CSV, default_outdir="model_comparison_marmoset"),
    # Humans do a single "test" session per subject (no p_success stages); the
    # cascade p_R only matters for PPC simulation, not ELPD (which re-uses the
    # observed pulses). Compares the AR vs non-AR human models.
    "human": dict(
        load_fn=load_human_sessions_compare,
        stage_p={"test": 0.7},
        stage_col="stage_cat", specie_filter="Human",
        default_stages="test",
        default_models="lapse_noleak_ar_human,lapse_noleak_human",
        default_data=_ALLSPECIES_CSV, default_outdir="model_comparison_humans"),
}
if CFG_SPECIES not in _SPECIES_ADAPTERS:
    raise RuntimeError(
        f"SPECIES={CFG_SPECIES!r} not supported for model comparison "
        f"(options: {sorted(_SPECIES_ADAPTERS)}).")
_ADAPTER = _SPECIES_ADAPTERS[CFG_SPECIES]
LOAD_FN = _ADAPTER["load_fn"]
STAGE_P = _ADAPTER["stage_p"]
STAGE_COL = _ADAPTER["stage_col"]
SPECIE_FILTER = _ADAPTER["specie_filter"]

MODELS     = [m.strip() for m in os.environ.get("MODELS", _ADAPTER["default_models"]).split(",") if m.strip()]
STAGES     = [s.strip() for s in os.environ.get("STAGES", _ADAPTER["default_stages"]).split(",") if s.strip()]
ANIMALS    = os.environ.get("ANIMALS", "").strip()
OUTDIR     = os.environ.get("OUTDIR", _ADAPTER["default_outdir"])
MODEL_DIR  = os.path.expanduser(os.environ.get("MODEL_DIR", "models"))
DATA_PATH  = os.environ.get("DATA_PATH", _ADAPTER["default_data"])
SEED       = int(os.environ.get("SEED", "0"))
DO_ELPD    = int(os.environ.get("DO_ELPD", "1"))
DO_PPC     = int(os.environ.get("DO_PPC", "1"))
ELPD_DRAWS = int(os.environ.get("ELPD_DRAWS", "100"))
ELPD_REPS  = int(os.environ.get("ELPD_REPLICATES", "300"))
ELPD_MAXT  = int(os.environ.get("ELPD_MAX_TRIALS", "1000"))
ELPD_METHOD = os.environ.get("ELPD_METHOD", "cv").strip().lower()        # "cv" (held-out) or "psis"
ELPD_FOLDS  = int(os.environ.get("ELPD_FOLDS", "5"))                     # K for cv
ELPD_RT_DENSITY = os.environ.get("ELPD_RT_DENSITY", "lognormal").strip().lower()
N_PPC      = int(os.environ.get("N_PPC", "200"))
N_POST     = int(os.environ.get("N_POST", "2000"))


def _default_model_file(model_name: str) -> str:
    values = "_".join(f"{int(round(v * 100)):03d}" for v in cfg.P_SUCCESS_TRAIN_VALUES)
    return f"npe_{model_name}_mixed_{values}.pt"


class LoadedModel:
    def __init__(self, name, device):
        self.name = name
        self.sim_fn, prior, self.param_names, _tag, self.ar = select_model(name)
        if hasattr(prior, "to"):
            prior.to(device)
        self.prior = prior
        path = os.path.join(MODEL_DIR, os.environ.get(f"MODEL_FILE_{name}", _default_model_file(name)))
        de, emb, saved_cfg, _T = load_npe_decoupled(path, prior_theta=prior, device=device)
        if bool(getattr(saved_cfg, "AUTOREGRESSIVE", False)) != self.ar:
            raise RuntimeError(f"{name}: AR mismatch vs checkpoint {path}")
        sp = getattr(saved_cfg, "SPECIES", None)
        if sp is not None and sp != CFG_SPECIES:
            raise RuntimeError(f"{name}: checkpoint SPECIES={sp!r} != {CFG_SPECIES!r} "
                               f"(set SPECIES={sp} to compare these checkpoints)")
        self.embedding = emb
        self.posterior = DirectPosterior(posterior_estimator=de, prior=prior, device=device)
        self.path = path
        print(f"  loaded {name} (ar={self.ar}) from {path}")


@torch.no_grad()
def _ppc_raw(model: LoadedModel, samples: np.ndarray, T: int, P: int,
             p_success: float, n_ppc: int, device) -> dict:
    """Posterior-predictive raw arrays (n_ppc, T): rt, choice, hit, net evidence,
    correct_side, and AR previous-trial sequences (when the model is AR)."""
    dev = torch.device(device)
    idx = np.random.choice(len(samples), size=min(n_ppc, len(samples)),
                           replace=len(samples) < n_ppc)
    theta = torch.tensor(samples[idx], dtype=torch.float32, device=dev)
    session_fn = simulate_training_sessions_ar if model.ar else simulate_training_sessions
    _, x_flat = session_fn(
        prior_theta=None, num_sessions=len(idx), num_trials=T,
        simulate_batch_fn=model.sim_fn, device=dev, mu_sensory=float(cfg.MU_SENSORY),
        p_success=float(p_success), P=P, log_rt=bool(cfg.LOG_RT_MANUALLY),
        seed=int(np.random.randint(0, 2**31 - 1)), theta=theta, warn_on_timeouts=False)
    td = trial_feature_dim(P, ar=model.ar)
    x3 = x_flat.view(len(idx), T, td).cpu().numpy()
    log_rt = x3[:, :, 0]; choice = x3[:, :, 1].astype(int)
    pulses = x3[:, :, (4 if model.ar else 2):]
    rt = np.exp(log_rt) if cfg.LOG_RT_MANUALLY else log_rt
    hit = rt < float(T_MAX) - 0.01
    out = dict(rt=rt, choice=choice, hit=hit,
               correct_side=(pulses.sum(-1) > 0).astype(int),
               net_evidence=pulses.sum(-1).astype(np.float32))
    if model.ar:
        out["choice_signed"] = (2 * choice - 1).astype(np.float32)
        out["prev_choice"] = x3[:, :, 2].astype(np.float32)
        out["prev_outcome"] = x3[:, :, 3].astype(np.float32)
    return out


def analyse_cell(animal, group, stage, models, device, rng) -> list[dict]:
    P = max_num_pulses()
    p_success = STAGE_P.get(stage, float(cfg.P_SUCCESS))
    rows = []
    sessions_cache = {}

    for model in models:
        if model.ar not in sessions_cache:
            sess, _m = LOAD_FN(csv_path=DATA_PATH, animal=animal, stage=stage,
                               log_rt=bool(cfg.LOG_RT_MANUALLY), seed=SEED,
                               autoregressive=model.ar)
            sessions_cache[model.ar] = sess
        sessions = sessions_cache[model.ar]
        if not sessions:
            return rows
        td = trial_feature_dim(P, ar=model.ar)
        all_trials = torch.cat([x.reshape(-1, td) for x in sessions], dim=0)
        n_trials = all_trials.shape[0]
        x_combined = all_trials.reshape(1, -1).to(device, dtype=torch.float32)

        with torch.no_grad():
            x_emb = model.embedding(x_combined)
        samples = model.posterior.sample((N_POST,), x=x_emb, show_progress_bars=False).cpu().numpy()

        row = dict(animal=animal, group=group, stage=stage, model=model.name,
                   n_trials=int(n_trials))

        if DO_PPC:
            try:
                obs = extract_observed(x_combined.cpu(), P, autoregressive=model.ar, device=torch.device("cpu"))
                rt = np.exp(obs.log_rt.numpy()); ch = obs.choice.numpy().astype(int)
                cs = obs.correct_side.numpy().astype(int); hit = rt < float(T_MAX) - 0.01
                net = obs.pulses.cpu().numpy().sum(axis=1)
                obs_summary = _session_summary(rt, ch, cs, hit)
                sim = _ppc_raw(model, samples, int(cfg.NUM_TRIALS_OBS), P, p_success, N_PPC, device)
                ppc = np.stack([_session_summary(sim["rt"][i], sim["choice"][i],
                                                 sim["correct_side"][i], sim["hit"][i])
                                for i in range(sim["rt"].shape[0])])
                d = ppc_discrepancy(obs_summary, ppc)
                row["ppc_discrepancy_z"] = d.discrepancy_z
                row["ppc_bayes_p"] = d.bayes_p_extreme

                # (1) RT distribution divergence: real vs pooled PPC
                rtd = rt_distribution_distance(rt[hit], sim["rt"].reshape(-1)[sim["hit"].reshape(-1)])
                row["rt_wasserstein_log"] = rtd["wasserstein_log"]
                row["rt_ks"] = rtd["ks"]
                # (2) psychometric slope mismatch is captured via the per-bin acc in
                #     ppc; (3) autoregressive structure: real vs PPC win-stay/lose-shift
                ar_real = ar_choice_stats(
                    obs.choice.cpu().numpy() * 2 - 1,
                    obs.prev_choice.cpu().numpy(), obs.prev_outcome.cpu().numpy(),
                ) if model.ar else dict(wsls_index=np.nan, lag1_choice_autocorr=np.nan)
                row["wsls_real"] = ar_real["wsls_index"]
                row["lag1_real"] = ar_real["lag1_choice_autocorr"]
                if model.ar and "prev_choice" in sim:
                    ar_sim = ar_stats_from_dict(sim, pooled=True)
                    row["wsls_sim"] = ar_sim["wsls_index"]
                    row["lag1_sim"] = ar_sim["lag1_choice_autocorr"]
                    row["wsls_abs_err"] = abs(ar_real["wsls_index"] - ar_sim["wsls_index"]) \
                        if np.isfinite(ar_real["wsls_index"]) and np.isfinite(ar_sim["wsls_index"]) else np.nan
            except Exception:
                print(f"    [warn] PPC failed {animal}/{stage}/{model.name}"); traceback.print_exc()
                row["ppc_discrepancy_z"] = np.nan; row["ppc_bayes_p"] = np.nan

        if DO_ELPD:
            try:
                obs = extract_observed(x_combined, P, autoregressive=model.ar, device=device)
                # subsample trials for tractability
                if obs.n_trials > ELPD_MAXT:
                    sel = torch.from_numpy(rng.choice(obs.n_trials, size=ELPD_MAXT, replace=False)).to(device)
                    obs = type(obs)(obs.log_rt[sel], obs.choice[sel], obs.pulses[sel],
                                    obs.prev_choice[sel], obs.prev_outcome[sel], obs.correct_side[sel])
                g = torch.Generator(device=device); g.manual_seed(SEED + 1)
                if ELPD_METHOD == "cv":
                    # Honest held-out CV: re-condition the amortized posterior on the
                    # train fold each split — no importance sampling, no Pareto-k.
                    res = crossval_elpd(
                        model.embedding, model.posterior, model.sim_fn, obs,
                        autoregressive=model.ar, mu_sensory=float(cfg.MU_SENSORY),
                        n_folds=ELPD_FOLDS, n_draws=ELPD_DRAWS, n_replicates=ELPD_REPS,
                        rt_density=ELPD_RT_DENSITY, device=device, generator=g, rng=rng)
                else:
                    didx = rng.choice(len(samples), size=min(ELPD_DRAWS, len(samples)),
                                      replace=len(samples) < ELPD_DRAWS)
                    theta_draws = torch.tensor(samples[didx], dtype=torch.float32, device=device)
                    ll = synthetic_loglik_matrix(model.sim_fn, theta_draws, obs, autoregressive=model.ar,
                                                 mu_sensory=float(cfg.MU_SENSORY), n_replicates=ELPD_REPS,
                                                 device=device, generator=g)
                    res = elpd_psis_loo(ll)
                # normalise by n_obs so cells with different trial counts are comparable
                row.update(elpd_loo=res["elpd_loo"], elpd_se=res["se"], p_loo=res["p_loo"],
                           pareto_k_max=res["pareto_k_max"], frac_k_gt_0_7=res["frac_k_gt_0_7"],
                           n_obs_elpd=res["n_obs"], elpd_per_trial=res["elpd_loo"] / max(1, res["n_obs"]),
                           elpd_method=res["method"], n_folds_elpd=res.get("n_folds", np.nan))
            except Exception:
                print(f"    [warn] ELPD failed {animal}/{stage}/{model.name}"); traceback.print_exc()
                row.update(elpd_loo=np.nan, elpd_se=np.nan, elpd_per_trial=np.nan)
        rows.append(row)

    # selection within this cell (by elpd_loo; models share the same trials)
    if DO_ELPD and len(rows) >= 2 and all("elpd_loo" in r for r in rows):
        valid = [r for r in rows if np.isfinite(r.get("elpd_loo", np.nan))]
        if len(valid) >= 2:
            valid.sort(key=lambda r: r["elpd_loo"], reverse=True)
            best, runner = valid[0], valid[1]
            diff = best["elpd_loo"] - runner["elpd_loo"]
            diff_se = float(np.hypot(best.get("elpd_se", 0.0), runner.get("elpd_se", 0.0)))
            for r in rows:
                r["selected"] = (r["model"] == best["model"])
                r["elpd_diff_to_best"] = r["elpd_loo"] - best["elpd_loo"] if np.isfinite(r.get("elpd_loo", np.nan)) else np.nan
            for r in rows:
                r["winner_model"] = best["model"]
                r["winner_elpd_diff"] = diff
                r["winner_elpd_diff_se"] = diff_se
                r["winner_decisive"] = bool(diff > 2 * diff_se)  # ~2 SE rule of thumb
    return rows


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Species: {CFG_SPECIES}  (P={max_num_pulses()}, T_MAX={T_MAX})")
    print(f"Comparing models: {MODELS}")
    print(f"ELPD: method={ELPD_METHOD} draws={ELPD_DRAWS} reps={ELPD_REPS} "
          f"max_trials={ELPD_MAXT} folds={ELPD_FOLDS} rt_density={ELPD_RT_DENSITY} "
          f"(do_elpd={DO_ELPD}, do_ppc={DO_PPC})")
    print(f"Data: {DATA_PATH}\nDevice: {device}\n")

    models = [LoadedModel(n, device) for n in MODELS]

    df_all = pd.read_csv(DATA_PATH, compression="infer",
                         dtype={"flashes_left": str, "flashes_right": str}, low_memory=False)
    if SPECIE_FILTER is not None:
        df_all = df_all[df_all["specie"] == SPECIE_FILTER]
    df_all = df_all[df_all[STAGE_COL].isin(STAGES)]
    if ANIMALS:
        wanted = {a.strip() for a in ANIMALS.split(",") if a.strip()}
        df_all = df_all[df_all["name"].isin(wanted)]
    roster = (df_all[["name", "group", STAGE_COL]].drop_duplicates()
              .rename(columns={STAGE_COL: "stage"})
              .sort_values(["group", "name", "stage"]).reset_index(drop=True))
    print(f"\n{len(roster)} (animal, stage) cells across {roster['name'].nunique()} mice.\n")

    all_rows = []
    for _, r in roster.iterrows():
        try:
            rows = analyse_cell(r["name"], r["group"], r["stage"], models, device, rng)
        except Exception:
            print(f"  [err] {r['name']}/{r['stage']}:"); traceback.print_exc(); continue
        for row in rows:
            msg = f"  {row['animal']:12s} {row['stage']:6s} {row['model']:24s}"
            if "elpd_loo" in row:
                msg += f" elpd={row['elpd_loo']:.1f} (/{row['n_trials']}tr={row.get('elpd_per_trial',float('nan')):.3f})"
            if "ppc_discrepancy_z" in row:
                msg += f" ppc_z={row['ppc_discrepancy_z']:.2f}"
            if row.get("selected"):
                msg += "  <== selected"
            print(msg)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No cells analysed.")

    df = pd.DataFrame(all_rows)
    per_cell = os.path.join(OUTDIR, "per_cell_comparison.csv")
    df.to_csv(per_cell, index=False)
    print(f"\nSaved {per_cell}")

    # --- group-level selection summary ---
    if "selected" in df.columns:
        sel = df[df["selected"] == True][["animal", "group", "stage", "winner_model",
                                          "winner_elpd_diff", "winner_elpd_diff_se", "winner_decisive"]]
        sel.to_csv(os.path.join(OUTDIR, "selection_per_cell.csv"), index=False)
        win_counts = (df[df["selected"] == True].groupby(["group", "winner_model"]).size()
                      .rename("n_cells_won").reset_index())
        win_counts.to_csv(os.path.join(OUTDIR, "win_counts_by_group.csv"), index=False)
        print("\n=== Model wins by genotype ===")
        print(win_counts.to_string(index=False))

        # mean elpd-per-trial by model × group
        if "elpd_per_trial" in df.columns:
            tbl = (df.groupby(["group", "model"])["elpd_per_trial"]
                   .agg(["mean", "std", "count"]).reset_index())
            tbl.to_csv(os.path.join(OUTDIR, "elpd_per_trial_by_group.csv"), index=False)
            print("\n=== Mean ELPD/trial by genotype × model ===")
            print(tbl.to_string(index=False))
            _plot_elpd(df, os.path.join(OUTDIR, "elpd_per_trial_by_model.png"))

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(dict(species=CFG_SPECIES, models=MODELS, model_files={m.name: m.path for m in models},
                       stages=STAGES, elpd_method=ELPD_METHOD, elpd_folds=ELPD_FOLDS,
                       elpd_rt_density=ELPD_RT_DENSITY, elpd_draws=ELPD_DRAWS, elpd_replicates=ELPD_REPS,
                       elpd_max_trials=ELPD_MAXT, n_ppc=N_PPC, n_post=N_POST,
                       n_cells=int(df.groupby(["animal", "stage"]).ngroups)), f, indent=2)
    print(f"\nAll done. Outputs in {OUTDIR}")


def _plot_elpd(df: pd.DataFrame, outpath: str):
    groups = sorted(df["group"].dropna().unique())
    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(groups), 4.5))
    width = 0.8 / max(1, len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for j, m in enumerate(models):
        means, sems, xs = [], [], []
        for i, g in enumerate(groups):
            vals = df[(df["group"] == g) & (df["model"] == m)]["elpd_per_trial"].dropna().values
            if len(vals) == 0:
                continue
            xs.append(i + (j - (len(models) - 1) / 2) * width)
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        ax.bar(xs, means, width=width, yerr=sems, capsize=3, label=m, color=colors[j], alpha=0.85)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=15)
    ax.set_ylabel("ELPD per trial (higher = better)")
    ax.set_title(f"Held-out predictive fit by genotype × model ({ELPD_METHOD})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
