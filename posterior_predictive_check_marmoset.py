"""
Posterior predictive check for the marmoset fit.

For each session:
  1. Load the real observed data (RT, choice) and the fitted posterior samples.
  2. Draw N_PPC parameter sets from the posterior.
  3. Simulate a synthetic session for each draw.
  4. Overlay simulated vs real RT histograms, accuracy, and timeout rate.

Usage
-----
  python posterior_predictive_check_marmoset.py
  ANIMAL=Helios STAGE=70-30 N_PPC=200 python posterior_predictive_check_marmoset.py

Outputs go to  marmoset_outputs/<ANIMAL>_<STAGE>/ppc/
"""
from __future__ import annotations

import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi_for_diffusion_models.models.lapse_rt_choice_model import simulate_rt_choice_batch_lapse
from sbi_for_diffusion_models.models.rt_choice_model import max_num_pulses
from sbi_for_diffusion_models.load_marmoset import load_marmoset_sessions
from sbi_for_diffusion_models.run_config import (
    RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL,
)

cfg = RUN_CONFIG_PARAMS

ANIMAL  = os.environ.get("ANIMAL", "Helios")
STAGE   = os.environ.get("STAGE",  "70-30")
N_PPC   = int(os.environ.get("N_PPC",   "200"))   # posterior draws to simulate
INDIR   = os.environ.get("INDIR",  f"marmoset_outputs/{ANIMAL}_{STAGE.replace('-','_')}")
OUTDIR  = os.environ.get("OUTDIR", os.path.join(INDIR, "ppc"))
DATA_PATH = "/projectnb/ssmsvi/rsenne/data_marmoset/marmoset_data.csv.gz"

PARAM_NAMES = ["a0", "lam", "v", "B", "tau", "p_lapse"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_from_posterior(posterior_samples: np.ndarray, n_ppc: int, T: int, P: int,
                             device: str = "cpu") -> dict:
    """
    Draw n_ppc parameter vectors from the posterior samples, simulate one
    session each, and return pooled RT / choice arrays.
    """
    dev = torch.device(device)
    idx = np.random.choice(len(posterior_samples), size=n_ppc, replace=len(posterior_samples) < n_ppc)
    theta = torch.tensor(posterior_samples[idx], dtype=torch.float32, device=dev)  # (n_ppc, 6)

    # Simulate T trials per draw (each row of theta is one parameter set)
    # We simulate T trials per draw → total n_ppc * T trials; then reshape.
    # For efficiency, simulate each draw separately (theta is per-session, not per-trial).
    all_rt, all_choice, all_hit, all_correct_side = [], [], [], []
    for i in range(n_ppc):
        th_i = theta[i].unsqueeze(0).expand(T, -1)   # (T, 6) same params for all trials
        # generate random pulses for this synthetic session
        pulses, correct_side_i = _random_pulses(T, P, p_success=float(cfg.P_SUCCESS), dev=dev)
        x_raw_i, hit_i, _ = simulate_rt_choice_batch_lapse(
            th_i,
            pulse_sides=pulses,
            mu_sensory=float(cfg.MU_SENSORY),
            p_success=float(cfg.P_SUCCESS),
        )
        # x_raw_i: (T, 2) — columns are [rt, choice]
        # choice is 1 if upper boundary crossed, 0 if lower boundary crossed.
        # correct = upper boundary when correct_side=+1, lower boundary when correct_side=-1.
        correct_choice = ((correct_side_i > 0).long()).cpu().numpy().astype(np.int32)
        all_rt.append(x_raw_i[:, 0].cpu().numpy().astype(np.float32))
        all_choice.append(x_raw_i[:, 1].cpu().numpy().astype(np.int32))
        all_hit.append(hit_i.cpu().numpy().astype(bool))
        all_correct_side.append(correct_choice)

    rt           = np.stack(all_rt)           # (n_ppc, T)
    choice       = np.stack(all_choice)       # (n_ppc, T)
    hit          = np.stack(all_hit)          # (n_ppc, T)
    correct_side = np.stack(all_correct_side) # (n_ppc, T) — 1 if correct=upper, 0 if correct=lower
    # per-draw summary stats (avoids tricky 2-D boolean indexing later)
    # accuracy = fraction of non-timeout choices that matched the correct boundary
    acc_per_draw     = np.array(
        [(choice[i][hit[i]] == correct_side[i][hit[i]]).mean() if hit[i].any() else np.nan
         for i in range(n_ppc)]
    )
    timeout_per_draw = 1.0 - hit.mean(axis=1)          # (n_ppc,)
    med_rt_per_draw  = np.array([np.median(rt[i][hit[i]]) if hit[i].any() else np.nan
                                  for i in range(n_ppc)])
    return {
        "rt":              rt,
        "choice":          choice,
        "hit":             hit,
        "acc_per_draw":     acc_per_draw,
        "timeout_per_draw": timeout_per_draw,
        "med_rt_per_draw":  med_rt_per_draw,
    }


def _random_pulses(T: int, P: int, p_success: float, dev: torch.device):
    """Generate random ±1 pulse sequences for T trials, each length P.
    Returns (pulses, correct_side) where correct_side is +1/-1 per trial."""
    correct_side = (torch.randint(0, 2, (T,), device=dev) * 2 - 1).float()  # ±1
    is_correct = (torch.rand(T, P, device=dev) < p_success)
    pulses = torch.where(
        is_correct,
        correct_side.unsqueeze(1).expand(T, P),
        -correct_side.unsqueeze(1).expand(T, P),
    )
    return pulses, correct_side  # (T, P), (T,)


def _extract_real(x_flat: torch.Tensor, T: int, P: int) -> dict:
    """Unpack a flat session tensor into RT and choice arrays."""
    x = x_flat.reshape(T, 2 + P).numpy()
    log_rt   = x[:, 0]
    choice   = x[:, 1].astype(int)
    # un-log RT; T_MAX marks timeout trials
    rt = np.exp(log_rt)
    hit = rt < float(T_MAX) - 0.01
    return {"rt": rt, "choice": choice, "hit": hit}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ppc_session(real: dict, sim: dict, session_label: str,
                     acc_real: float, outpath: str):
    """4-panel PPC plot for one session."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"Posterior Predictive Check — {session_label}  (acc={acc_real:.1%})",
                 fontsize=12)

    rt_max = float(T_MAX)
    bins = np.linspace(0, rt_max, 41)

    # ---- panel 1: RT histogram ----
    ax = axes[0]
    real_rt_hit = real["rt"][real["hit"]]
    sim_rt_hit  = sim["rt"].reshape(-1)[sim["hit"].reshape(-1)]  # pool all draws

    ax.hist(real_rt_hit, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="Real")
    ax.hist(sim_rt_hit,  bins=bins, density=True, alpha=0.4,
            color="tomato", label=f"PPC (n={N_PPC})")
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Density")
    ax.set_title("RT distribution (non-timeout)")
    ax.legend(fontsize=8)

    # ---- panel 2: cumulative RT ----
    ax = axes[1]
    for rt_arr, col, lbl in [(real_rt_hit, "steelblue", "Real"),
                              (sim_rt_hit,  "tomato",    "PPC")]:
        s = np.sort(rt_arr)
        ax.plot(s, np.linspace(0, 1, len(s)), color=col, label=lbl, alpha=0.8)
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("Cumulative RT")
    ax.legend(fontsize=8)

    # ---- panel 3: accuracy distribution across PPC draws ----
    ax = axes[2]
    # acc_real comes from session metadata (choice == correct_side), not from mean(choice)
    # which would give ~0.5 under absolute right/left encoding.
    ax.hist(sim["acc_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC draws")
    ax.axvline(acc_real, color="steelblue", lw=2, label=f"Real ({acc_real:.2f})")
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy distribution")
    ax.legend(fontsize=8)

    # ---- panel 4: timeout rate ----
    ax = axes[3]
    real_timeout = 1.0 - real["hit"].mean()
    ax.hist(sim["timeout_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC draws")
    ax.axvline(real_timeout, color="steelblue", lw=2,
               label=f"Real ({real_timeout:.2f})")
    ax.set_xlabel("Timeout rate")
    ax.set_title("Timeout rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cpu"   # PPC runs fine on CPU
    os.makedirs(OUTDIR, exist_ok=True)

    P = max_num_pulses()
    T = int(cfg.NUM_TRIALS_OBS)
    print(f"Animal={ANIMAL}  Stage={STAGE}  N_PPC={N_PPC}  P={P}  T={T}")
    print(f"Reading posteriors from: {INDIR}")
    print(f"Writing PPC plots to:    {OUTDIR}\n")

    # Load real sessions
    sessions, meta = load_marmoset_sessions(
        csv_path=DATA_PATH,
        animal=ANIMAL,
        stage=STAGE,
        num_trials_per_session=T,
        p_max=P,
        log_rt=bool(cfg.LOG_RT_MANUALLY),
    )
    if not sessions:
        raise RuntimeError(f"No sessions found for {ANIMAL}/{STAGE}")

    # Find matching posterior files (sorted by session index)
    post_files = sorted(glob.glob(os.path.join(INDIR, "posterior_sess*.npy")))
    if len(post_files) != len(sessions):
        raise RuntimeError(
            f"Found {len(post_files)} posterior files but {len(sessions)} sessions."
        )

    summary_rows = []

    for s_idx, (x_flat, m, post_file) in enumerate(zip(sessions, meta, post_files)):
        label = str(m["session_datetime"]).replace(" ", "_").replace(":", "-")
        acc   = float(m["accuracy"])
        print(f"[Session {s_idx+1}/{len(sessions)}]  {label}  acc={acc:.3f}")

        # Real data
        real = _extract_real(x_flat, T, P)

        # Posterior samples
        post_samples = np.load(post_file)   # (N_post, 6)
        print(f"  Posterior samples: {post_samples.shape}  "
              f"  v mean={post_samples[:,2].mean():.3f}  tau mean={post_samples[:,4].mean():.3f}")

        # Simulate from posterior
        sim = simulate_from_posterior(post_samples, N_PPC, T, P, device=device)

        # Summary stats — use metadata accuracy (choice==correct_side), not mean(choice)
        real_acc     = acc
        sim_acc_med  = float(np.nanmedian(sim["acc_per_draw"]))
        real_med_rt  = float(np.median(real["rt"][real["hit"]])) if real["hit"].any() else float("nan")
        sim_med_rt   = float(np.nanmedian(sim["med_rt_per_draw"]))
        real_timeout = 1.0 - real["hit"].mean()
        sim_timeout  = float(np.mean(sim["timeout_per_draw"]))

        print(f"  RT median  — real: {real_med_rt:.3f}s   sim: {sim_med_rt:.3f}s")
        print(f"  Accuracy   — real: {real_acc:.3f}      sim: {sim_acc_med:.3f}")
        print(f"  Timeout    — real: {real_timeout:.3f}      sim: {sim_timeout:.3f}")

        summary_rows.append({
            "session": label, "acc_real": real_acc, "acc_sim": sim_acc_med,
            "rt_med_real": real_med_rt, "rt_med_sim": sim_med_rt,
            "timeout_real": real_timeout, "timeout_sim": sim_timeout,
        })

        # Plot
        plot_ppc_session(
            real, sim, label, acc,
            outpath=os.path.join(OUTDIR, f"ppc_sess{s_idx+1}_{label}.png"),
        )

    # Summary table
    print("\n=== PPC Summary ===")
    print(f"{'Session':35s}  {'acc_real':>8s}  {'acc_sim':>8s}  {'rt_real':>8s}  {'rt_sim':>8s}  {'to_real':>8s}  {'to_sim':>8s}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r['session']:35s}  {r['acc_real']:8.3f}  {r['acc_sim']:8.3f}  "
              f"{r['rt_med_real']:8.3f}  {r['rt_med_sim']:8.3f}  "
              f"{r['timeout_real']:8.3f}  {r['timeout_sim']:8.3f}")

    summary_path = os.path.join(OUTDIR, "ppc_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Posterior Predictive Check Summary\n")
        f.write(f"Animal={ANIMAL}  Stage={STAGE}  N_PPC={N_PPC}\n\n")
        f.write(f"{'Session':35s}  {'acc_real':>8s}  {'acc_sim':>8s}  "
                f"{'rt_real':>8s}  {'rt_sim':>8s}  {'to_real':>8s}  {'to_sim':>8s}\n")
        f.write("-" * 95 + "\n")
        for r in summary_rows:
            f.write(f"{r['session']:35s}  {r['acc_real']:8.3f}  {r['acc_sim']:8.3f}  "
                    f"{r['rt_med_real']:8.3f}  {r['rt_med_sim']:8.3f}  "
                    f"{r['timeout_real']:8.3f}  {r['timeout_sim']:8.3f}\n")
    print(f"\nSaved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
