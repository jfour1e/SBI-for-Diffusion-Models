"""Posterior-predictive diagnostics shared by the fit and comparison scripts.

Three families of checks (per the analysis plan):

1. **RT distribution** — simulated vs real, quantified by a distributional
   distance (Wasserstein-1 on log-RT, robust to the heavy RT tail; plus the
   two-sample KS statistic). Lower = better.

2. **Choice accuracy / psychometric** — does the model predict the right
   accuracy, and the right *shape* of P(choose right) as a function of the net
   pulse evidence (the psychometric curve)?

3. **Autoregressive structure** — does the data show sensible sequential
   dependencies (win-stay/lose-shift, lag-1 choice autocorrelation), and does
   the model reproduce them? An iid (non-AR) model predicts no dependency
   (P(stay)≈0.5, autocorr≈0), so this is the key check that distinguishes the
   AR model from the non-AR one.

All functions are framework-light (numpy + scipy + matplotlib) and operate on
already-extracted per-trial arrays, so they work for any species.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.stats import wasserstein_distance, ks_2samp
except Exception:  # pragma: no cover
    wasserstein_distance = None
    ks_2samp = None


# ---------------------------------------------------------------------------
# 1. RT distribution distance
# ---------------------------------------------------------------------------

def rt_distribution_distance(real_rt: np.ndarray, sim_rt: np.ndarray) -> dict:
    """Distance between real and simulated RT distributions (non-timeout only).

    Returns Wasserstein-1 on RT and on log-RT, plus the KS statistic. NaNs if
    either sample is empty or scipy is unavailable.
    """
    real = np.asarray(real_rt, dtype=float)
    sim = np.asarray(sim_rt, dtype=float)
    real = real[np.isfinite(real) & (real > 0)]
    sim = sim[np.isfinite(sim) & (sim > 0)]
    if real.size == 0 or sim.size == 0 or wasserstein_distance is None:
        return dict(wasserstein=np.nan, wasserstein_log=np.nan, ks=np.nan)
    return dict(
        wasserstein=float(wasserstein_distance(real, sim)),
        wasserstein_log=float(wasserstein_distance(np.log(real), np.log(sim))),
        ks=float(ks_2samp(real, sim).statistic),
    )


# ---------------------------------------------------------------------------
# 2. Psychometric: P(choose right) vs net pulse evidence
# ---------------------------------------------------------------------------

def psychometric_curve(net_evidence: np.ndarray, choice_right: np.ndarray,
                       *, bins: Optional[np.ndarray] = None, n_bins: int = 11) -> tuple:
    """Return (bin_centers, p_right, counts) for P(choose right) vs net evidence.

    `net_evidence` is the signed pulse sum per trial (>0 favours right);
    `choice_right` is 0/1. Bins are symmetric around 0 unless provided.
    """
    net = np.asarray(net_evidence, dtype=float)
    ch = np.asarray(choice_right, dtype=float)
    ok = np.isfinite(net) & np.isfinite(ch)
    net, ch = net[ok], ch[ok]
    if net.size == 0:
        return np.array([]), np.array([]), np.array([])
    if bins is None:
        lim = np.percentile(np.abs(net), 98)
        lim = max(float(lim), 1.0)
        bins = np.linspace(-lim, lim, n_bins)
    idx = np.digitize(net, bins)
    centers, pr, n = [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 0:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            pr.append(float(ch[m].mean()))
            n.append(int(m.sum()))
    return np.array(centers), np.array(pr), np.array(n)


# ---------------------------------------------------------------------------
# 3. Autoregressive (sequential) choice structure
# ---------------------------------------------------------------------------

def ar_choice_stats(choice_signed: np.ndarray,
                    prev_choice_signed: np.ndarray,
                    prev_outcome_signed: np.ndarray) -> dict:
    """Win-stay/lose-shift summary from signed choice sequences.

    Inputs (M,) with values in {-1,0,+1}; `prev_choice_signed == 0` flags a
    session start (no previous trial), which is excluded.

    Returns:
      p_stay_after_correct : P(repeat previous choice | previous was correct)
      p_stay_after_error   : P(repeat previous choice | previous was error)
      wsls_index           : p_stay_after_correct - p_stay_after_error
                             (>0 = win-stay/lose-shift; 0 = no history effect)
      lag1_choice_autocorr : corr(choice_t, choice_{t-1}) over has-prev trials
    """
    cs = np.asarray(choice_signed, dtype=float)
    pc = np.asarray(prev_choice_signed, dtype=float)
    po = np.asarray(prev_outcome_signed, dtype=float)
    has_prev = pc != 0
    stay = (cs == pc).astype(float)
    after_corr = has_prev & (po > 0)
    after_err = has_prev & (po < 0)

    def _mean(mask):
        return float(stay[mask].mean()) if mask.sum() > 0 else np.nan

    p_corr = _mean(after_corr)
    p_err = _mean(after_err)
    if has_prev.sum() > 1:
        a, b = cs[has_prev], pc[has_prev]
        lag1 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    else:
        lag1 = np.nan
    wsls = (p_corr - p_err) if (np.isfinite(p_corr) and np.isfinite(p_err)) else np.nan
    return dict(p_stay_after_correct=p_corr, p_stay_after_error=p_err,
                wsls_index=wsls, lag1_choice_autocorr=lag1)


# ---------------------------------------------------------------------------
# Plotting: a 6-panel PPC figure covering all three checks
# ---------------------------------------------------------------------------

def plot_ppc_panels(real: dict, sim: dict, *, label: str, outpath: str,
                    t_max: float, ar: bool, n_ppc: int) -> dict:
    """Render the PPC figure and return the scalar diagnostics it computed.

    `real` keys: rt, choice, hit, correct_side, net_evidence
                 (+ choice_signed, prev_choice, prev_outcome when ar).
    `sim`  keys: rt, choice, hit, correct_side, net_evidence,
                 acc_per_draw, p_right_per_draw
                 (+ choice_signed, prev_choice, prev_outcome when ar);
                 all 2-D (n_ppc, T) except the *_per_draw vectors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhit = real["hit"]
    real_acc = float((real["choice"][rhit] == real["correct_side"][rhit]).mean()) if rhit.any() else np.nan
    real_rt_hit = real["rt"][rhit]
    sim_rt_hit = sim["rt"].reshape(-1)[sim["hit"].reshape(-1)]

    rt_dist = rt_distribution_distance(real_rt_hit, sim_rt_hit)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"PPC — {label}  (real acc={real_acc:.1%})", fontsize=13)

    # (1) RT histogram
    ax = axes[0, 0]
    bins = np.linspace(0, float(t_max), 41)
    ax.hist(real_rt_hit, bins=bins, density=True, alpha=0.6, color="steelblue", label="Real")
    ax.hist(sim_rt_hit, bins=bins, density=True, alpha=0.4, color="tomato", label=f"PPC (n={n_ppc})")
    ax.set_xlabel("RT (s)"); ax.set_ylabel("Density"); ax.set_title("RT distribution")
    ax.legend(fontsize=8)

    # (2) RT cumulative + distance annotation
    ax = axes[0, 1]
    for arr, col, lbl in [(real_rt_hit, "steelblue", "Real"), (sim_rt_hit, "tomato", "PPC")]:
        if len(arr):
            s = np.sort(arr); ax.plot(s, np.linspace(0, 1, len(s)), color=col, label=lbl, alpha=0.85)
    ax.set_xlabel("RT (s)"); ax.set_ylabel("Cumulative"); ax.set_title("RT CDF")
    ax.text(0.55, 0.05,
            f"W1(logRT)={rt_dist['wasserstein_log']:.3f}\nKS={rt_dist['ks']:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray"))
    ax.legend(fontsize=8)

    # (3) Accuracy: real vs PPC distribution
    ax = axes[0, 2]
    ax.hist(sim["acc_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_acc, color="steelblue", lw=2, label=f"Real ({real_acc:.2f})")
    ax.set_xlabel("Accuracy"); ax.set_title("Choice accuracy"); ax.legend(fontsize=8)

    # (4) Side preference P(right)
    ax = axes[1, 0]
    real_p_right = float(real["choice"][rhit].mean()) if rhit.any() else np.nan
    ax.hist(sim["p_right_per_draw"], bins=20, color="tomato", alpha=0.7, label="PPC")
    ax.axvline(real_p_right, color="steelblue", lw=2, label=f"Real ({real_p_right:.2f})")
    ax.set_xlabel("P(right)"); ax.set_title("Side preference"); ax.legend(fontsize=8)

    # (5) Psychometric: P(right) vs net evidence
    ax = axes[1, 1]
    rc, rp, _ = psychometric_curve(real["net_evidence"][rhit], real["choice"][rhit])
    sh = sim["hit"].reshape(-1)
    sc, sp, _ = psychometric_curve(sim["net_evidence"].reshape(-1)[sh],
                                   sim["choice"].reshape(-1)[sh])
    ax.plot(rc, rp, "o-", color="steelblue", label="Real")
    ax.plot(sc, sp, "s--", color="tomato", alpha=0.8, label="PPC")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Net pulse evidence (Σ s)"); ax.set_ylabel("P(choose right)")
    ax.set_title("Psychometric"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)

    # (6) Autoregressive structure: win-stay / lose-shift
    ax = axes[1, 2]
    ar_real = ar_stats_from_dict(real)
    out_diag = dict(rt_dist=rt_dist, ar_real=ar_real)
    if ar and ("prev_choice" in sim):
        ar_sim = ar_stats_from_dict(sim, pooled=True)
        out_diag["ar_sim"] = ar_sim
        labels = ["stay|correct", "stay|error", "WSLS idx", "lag1 ac"]
        rvals = [ar_real["p_stay_after_correct"], ar_real["p_stay_after_error"],
                 ar_real["wsls_index"], ar_real["lag1_choice_autocorr"]]
        svals = [ar_sim["p_stay_after_correct"], ar_sim["p_stay_after_error"],
                 ar_sim["wsls_index"], ar_sim["lag1_choice_autocorr"]]
        xpos = np.arange(len(labels))
        ax.bar(xpos - 0.2, rvals, width=0.4, color="steelblue", label="Real")
        ax.bar(xpos + 0.2, svals, width=0.4, color="tomato", alpha=0.8, label="PPC")
        ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_title("Autoregressive structure"); ax.legend(fontsize=8)
    else:
        labels = ["stay|correct", "stay|error", "WSLS idx", "lag1 ac"]
        rvals = [ar_real["p_stay_after_correct"], ar_real["p_stay_after_error"],
                 ar_real["wsls_index"], ar_real["lag1_choice_autocorr"]]
        ax.bar(np.arange(len(labels)), rvals, width=0.55, color="steelblue", label="Real")
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_title("Autoregressive structure (real)\n(iid model predicts stay≈0.5, ac≈0)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_diag


def ar_stats_from_dict(d: dict, *, pooled: bool = False) -> dict:
    """Compute ar_choice_stats from a real (1-D) or sim (2-D, pooled) dict."""
    if "prev_choice" not in d:
        return dict(p_stay_after_correct=np.nan, p_stay_after_error=np.nan,
                    wsls_index=np.nan, lag1_choice_autocorr=np.nan)
    if pooled:
        cs = d["choice_signed"].reshape(-1)
        pc = d["prev_choice"].reshape(-1)
        po = d["prev_outcome"].reshape(-1)
    else:
        cs = np.asarray(d["choice_signed"]); pc = np.asarray(d["prev_choice"]); po = np.asarray(d["prev_outcome"])
    return ar_choice_stats(cs, pc, po)
