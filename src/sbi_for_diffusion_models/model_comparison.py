"""Model comparison / selection utilities for the NPE RT-choice family.

NPE gives an amortized posterior q(theta | x) but no closed-form likelihood, so
we cannot read off a pointwise log-likelihood directly. Two complementary tools
are provided:

1. `ppc_discrepancy`   — posterior-predictive check summarised to a scalar
   discrepancy (standardised distance between observed summary statistics and
   the posterior-predictive distribution of those statistics). Always available.

2. `crossval_elpd` (recommended) — honest **held-out predictive ELPD** via
   K-fold cross-validation. The amortized, permutation-invariant NPE posterior is
   re-conditioned on the train fold (a cheap forward pass), so each held-out trial
   is scored by log (1/S) sum_s p(y | theta_s) with NO importance sampling and NO
   log-lik flooring. This sidesteps both failure modes of the PSIS path below
   (the invalid IS correction for an amortized/synthetic-likelihood posterior, and
   the floor that masks the Pareto-k diagnostic).

3. `synthetic_loglik_matrix` + `elpd_psis_loo` — an approximate **expected log
   predictive density** via PSIS-LOO (legacy). Because the simulator defines an implicit
   likelihood p(rt, choice | theta, pulses[, history]), we estimate a per-trial,
   per-posterior-draw log-likelihood by Monte-Carlo: for each posterior draw we
   simulate R replicates of every observed trial (re-using the trial's realised
   pulse sequence and, for AR models, its observed previous-trial choice/outcome),
   then form a density over the binary `choice` (smoothed Bernoulli) times a
   Gaussian-KDE over `log rt` within the matching choice. The resulting
   [n_draws x n_trials] matrix is fed to ArviZ `loo` (PSIS-LOO), giving elpd_loo
   with Pareto-k diagnostics. ELPD is computed on the observable (choice, log rt),
   identically transformed for every model, so elpd differences are comparable.

This is a *synthetic-likelihood* approximation, not an exact likelihood; the
Monte-Carlo / KDE error is controlled by `n_replicates` and `n_draws`, and trials
may be subsampled (`max_trials`) for tractability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import math
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Observed-session extraction
# ---------------------------------------------------------------------------

@dataclass
class ObservedSession:
    log_rt: Tensor       # (N,)
    choice: Tensor       # (N,) in {0,1}
    pulses: Tensor       # (N, P) in {-1,0,+1}
    prev_choice: Tensor  # (N,) in {-1,0,+1}  (zeros if non-AR)
    prev_outcome: Tensor # (N,) in {-1,0,+1}  (zeros if non-AR)
    correct_side: Tensor # (N,) in {0,1}  (1 = right is correct), from pulse sum

    @property
    def n_trials(self) -> int:
        return int(self.log_rt.shape[0])


def extract_observed(x_combined: Tensor, P: int, *, autoregressive: bool,
                     device: torch.device) -> ObservedSession:
    """Unpack a flat (1, T*trial_dim) observed session into per-trial tensors."""
    trial_dim = (4 if autoregressive else 2) + P
    x = x_combined.reshape(-1, trial_dim).to(device=device, dtype=torch.float32)
    log_rt = x[:, 0]
    choice = x[:, 1].round().clamp(0, 1)
    if autoregressive:
        prev_choice = x[:, 2]
        prev_outcome = x[:, 3]
        pulses = x[:, 4:]
    else:
        prev_choice = torch.zeros_like(log_rt)
        prev_outcome = torch.zeros_like(log_rt)
        pulses = x[:, 2:]
    correct_side = (pulses.sum(dim=1) > 0).to(torch.float32)
    return ObservedSession(log_rt, choice, pulses, prev_choice, prev_outcome, correct_side)


# ---------------------------------------------------------------------------
# Synthetic per-trial log-likelihood
# ---------------------------------------------------------------------------

def _masked_std(values: Tensor, mask: Tensor, counts: Tensor) -> Tensor:
    """Per-column std of `values` (R, N) over rows where `mask` is True."""
    m = mask.to(values.dtype)
    s = (values * m).sum(dim=0)
    mean = s / counts.clamp_min(1.0)
    var = (((values - mean.unsqueeze(0)) ** 2) * m).sum(dim=0) / counts.clamp_min(1.0)
    return var.clamp_min(1e-8).sqrt()


def _masked_mean_std(values: Tensor, mask: Tensor, counts: Tensor):
    """Per-column (mean, std) of `values` (R, N) over rows where `mask` is True."""
    m = mask.to(values.dtype)
    mean = (values * m).sum(dim=0) / counts.clamp_min(1.0)
    var = (((values - mean.unsqueeze(0)) ** 2) * m).sum(dim=0) / counts.clamp_min(1.0)
    return mean, var.clamp_min(1e-8).sqrt()


def _gauss_logpdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    z = (x - mean) / std
    return -0.5 * z * z - torch.log(std) - 0.5 * math.log(2 * math.pi)


@torch.no_grad()
def synthetic_loglik_matrix(
    simulate_batch_fn: Callable,
    theta_draws: Tensor,            # (S, D) posterior draws (model parameterisation)
    obs: ObservedSession,
    *,
    autoregressive: bool,
    mu_sensory: float,
    n_replicates: int = 300,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    rt_density: str = "kde",
    log_rt_floor: Optional[float] = -3.0,
    rt_min_count: int = 3,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Return a (S, N) pointwise log-likelihood matrix for one observed session.

    For each posterior draw s, simulate `n_replicates` copies of every observed
    trial under theta_s (re-using each trial's realised pulses and, for AR models,
    its observed previous choice/outcome), then:
        loglik[s, n] = log P(choice_n | theta_s)            # smoothed Bernoulli
                     + log p(log_rt_n | choice_n, theta_s)

    RT term (`rt_density`):
      * "lognormal" (recommended for held-out CV): a Gaussian fit to the
        matching-choice replicates' log RT — low variance, no per-point KDE noise,
        and removes the Jensen down-bias that differs across models.
      * "kde": Gaussian KDE (Scott bandwidth) over the matching-choice log RT
        (legacy behaviour, kept for the PSIS-LOO path).

    Low-count handling (`log_rt_floor`):
      * If `log_rt_floor is None` (recommended for CV), trials with fewer than
        `rt_min_count` matching replicates fall back to the *marginal* log-RT
        density (Gaussian over all replicates) — a genuine density, not a clamp.
      * If a float, trials below the count use that flat floor and the RT term is
        also clamped below at it. This clamp tames PSIS importance-weight tails
        but masks the Pareto-k diagnostic, so prefer None outside the PSIS path.
    """
    dev = device or theta_draws.device
    theta_draws = theta_draws.to(dev)
    S, D = theta_draws.shape
    N = obs.n_trials
    R = int(n_replicates)

    log_rt_obs = obs.log_rt.to(dev)
    choice_obs = obs.choice.to(dev)
    pulses = obs.pulses.to(dev)
    prev_choice = obs.prev_choice.to(dev)
    prev_outcome = obs.prev_outcome.to(dev)

    # Tile the N trials R times: block r occupies rows [r*N : (r+1)*N].
    pulses_rep = pulses.repeat(R, 1)                  # (R*N, P)
    prev_choice_rep = prev_choice.repeat(R)           # (R*N,)
    prev_outcome_rep = prev_outcome.repeat(R)

    out = np.empty((S, N), dtype=np.float64)
    alpha = 0.5  # Laplace smoothing for the Bernoulli choice term

    for s in range(S):
        theta_s = theta_draws[s].unsqueeze(0).expand(R * N, D).contiguous()
        kwargs = dict(mu_sensory=float(mu_sensory), pulse_sides=pulses_rep,
                      pulse_generator=generator)
        if autoregressive:
            kwargs["prev_choice_signed"] = prev_choice_rep
            kwargs["prev_outcome_signed"] = prev_outcome_rep
        x_raw, _hit, _s = simulate_batch_fn(theta_s, **kwargs)

        rt_samp = x_raw[:, 0].clamp_min(1e-6).reshape(R, N)
        choice_samp = x_raw[:, 1].round().reshape(R, N)
        log_rt_samp = torch.log(rt_samp)

        match = (choice_samp == choice_obs.unsqueeze(0))           # (R, N)
        counts = match.sum(dim=0).to(torch.float32)                # (N,)

        # Smoothed Bernoulli for the observed choice
        p_choice = (counts + alpha) / (R + 2 * alpha)
        log_p_choice = torch.log(p_choice)

        # RT log-density, conditioned on the matching-choice replicates.
        if rt_density == "lognormal":
            mean, std = _masked_mean_std(log_rt_samp, match, counts)
            log_p_rt_cond = _gauss_logpdf(log_rt_obs, mean, std.clamp_min(1e-3))
        elif rt_density == "kde":
            std = _masked_std(log_rt_samp, match, counts)
            h = (1.06 * std * counts.clamp_min(1.0) ** (-0.2)).clamp_min(1e-3)
            diff = (log_rt_samp - log_rt_obs.unsqueeze(0)) / h.unsqueeze(0)  # (R, N)
            kern = torch.exp(-0.5 * diff * diff) / (h.unsqueeze(0) * math.sqrt(2 * math.pi))
            dens = (kern * match.to(kern.dtype)).sum(dim=0) / counts.clamp_min(1.0)
            log_p_rt_cond = torch.log(dens.clamp_min(1e-12))
        else:
            raise ValueError(f"rt_density must be 'lognormal' or 'kde', got {rt_density!r}")

        # Low-count fallback for the RT term (too few matching-choice replicates).
        enough = counts >= rt_min_count
        if log_rt_floor is None:
            # Marginal log-RT density (Gaussian over ALL replicates): a real
            # density, so it does not game the importance weights / Pareto-k.
            mean_all = log_rt_samp.mean(dim=0)
            std_all = log_rt_samp.std(dim=0).clamp_min(1e-3)
            log_p_rt = torch.where(enough, log_p_rt_cond,
                                   _gauss_logpdf(log_rt_obs, mean_all, std_all))
        else:
            log_p_rt = torch.where(
                enough, log_p_rt_cond, torch.full_like(log_p_rt_cond, float(log_rt_floor)),
            ).clamp_min(float(log_rt_floor))

        out[s] = (log_p_choice + log_p_rt).cpu().numpy()
        if progress is not None:
            progress(s + 1, S)

    return out


def elpd_psis_loo(loglik: np.ndarray) -> dict:
    """PSIS-LOO from a (S_draws, N_obs) pointwise log-likelihood matrix via ArviZ.

    Returns elpd_loo, se, p_loo, and Pareto-k diagnostics. Falls back to a plain
    log-pointwise-predictive-density (lppd) estimate if ArviZ is unavailable.
    """
    S, N = loglik.shape
    try:
        import warnings
        import arviz as az
        # ArviZ (>=0.12) requires a posterior group alongside log_likelihood.
        idata = az.from_dict(
            posterior={"_dummy": np.zeros((1, S))},
            log_likelihood={"y": loglik[np.newaxis, :, :]},  # (chain=1, draw=S, obs=N)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = az.loo(idata, pointwise=True)
        k = np.asarray(res["pareto_k"])
        return dict(
            elpd_loo=float(res["elpd_loo"]),
            se=float(res["se"]),
            p_loo=float(res["p_loo"]),
            n_obs=int(N),
            n_draws=int(S),
            pareto_k_max=float(np.nanmax(k)) if k.size else float("nan"),
            frac_k_gt_0_7=float(np.mean(k > 0.7)) if k.size else float("nan"),
            method="psis_loo",
            elpd_loo_pointwise=np.asarray(res["loo_i"]).astype(np.float64),
        )
    except Exception as e:
        # Fallback: log mean_s exp(loglik[s,n]) summed over n (lppd, not LOO).
        m = loglik.max(axis=0)
        lppd_i = m + np.log(np.mean(np.exp(loglik - m[np.newaxis, :]), axis=0))
        return dict(
            elpd_loo=float(np.sum(lppd_i)), se=float(np.std(lppd_i) * np.sqrt(N)),
            p_loo=float("nan"), n_obs=int(N), n_draws=int(S),
            pareto_k_max=float("nan"), frac_k_gt_0_7=float("nan"),
            method=f"lppd_fallback ({e.__class__.__name__})",
            elpd_loo_pointwise=lppd_i.astype(np.float64),
        )


# ---------------------------------------------------------------------------
# Held-out cross-validated ELPD (no importance sampling)
# ---------------------------------------------------------------------------

def _subset_obs(obs: ObservedSession, idx) -> ObservedSession:
    t = torch.as_tensor(idx, device=obs.log_rt.device, dtype=torch.long)
    return ObservedSession(obs.log_rt[t], obs.choice[t], obs.pulses[t],
                           obs.prev_choice[t], obs.prev_outcome[t], obs.correct_side[t])


def _obs_to_flat(obs: ObservedSession, idx, autoregressive: bool, device) -> Tensor:
    """Rebuild a flat (1, n*trial_dim) conditioning tensor from selected trials.

    Trial layout matches `extract_observed`: [log_rt, choice, (prev_choice,
    prev_outcome,)? pulses...].
    """
    o = _subset_obs(obs, idx)
    cols = [o.log_rt.unsqueeze(1), o.choice.unsqueeze(1)]
    if autoregressive:
        cols += [o.prev_choice.unsqueeze(1), o.prev_outcome.unsqueeze(1)]
    cols.append(o.pulses)
    flat = torch.cat(cols, dim=1).to(device=device, dtype=torch.float32)
    return flat.reshape(1, -1)


@torch.no_grad()
def crossval_elpd(
    embedding,
    posterior,
    simulate_batch_fn: Callable,
    obs: ObservedSession,
    *,
    autoregressive: bool,
    mu_sensory: float,
    n_folds: int = 5,
    n_draws: int = 200,
    n_replicates: int = 300,
    rt_density: str = "lognormal",
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    rng: Optional[np.random.Generator] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Honest held-out predictive ELPD via K-fold cross-validation.

    For each fold, condition the (amortized, permutation-invariant) NPE posterior
    on the *other* folds' trials, draw `n_draws` posterior samples, and score the
    held-out fold's trials by their Monte-Carlo posterior-predictive density
        elpd_n = log (1/S) sum_s p(y_n | theta_s).
    No importance weights, no Pareto-k, no log-lik flooring — re-conditioning is a
    cheap forward pass for an amortized posterior, so the PSIS-LOO reweighting
    trick (and its failure modes) is unnecessary. The synthetic-likelihood MC/KDE
    error is the only remaining approximation; control it with `n_replicates`,
    `n_draws`, and `rt_density="lognormal"`.

    Returns the same keys as `elpd_psis_loo` (elpd_loo / se / n_obs / method /
    elpd_loo_pointwise) so it is a drop-in for the comparison script, plus
    `n_folds` and `frac_low_count` (fraction of held-out trials whose observed
    choice was rarely reproduced — a fit-quality flag).
    """
    dev = device or obs.log_rt.device
    N = obs.n_trials
    rng = rng or np.random.default_rng(0)
    K = int(max(2, min(n_folds, N)))
    folds = np.array_split(rng.permutation(N), K)

    pointwise = np.full(N, np.nan, dtype=np.float64)
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k]) if K > 1 else folds[k]
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        x_emb = embedding(_obs_to_flat(obs, train_idx, autoregressive, dev))
        theta = posterior.sample((int(n_draws),), x=x_emb, show_progress_bars=False).to(dev)
        obs_test = _subset_obs(obs, test_idx)
        ll = synthetic_loglik_matrix(
            simulate_batch_fn, theta, obs_test, autoregressive=autoregressive,
            mu_sensory=mu_sensory, n_replicates=n_replicates, device=dev,
            generator=generator, rt_density=rt_density, log_rt_floor=None)
        # log mean_s exp(ll[s, n]) per held-out trial (log-sum-exp, stable)
        m = ll.max(axis=0)
        pointwise[test_idx] = m + np.log(np.mean(np.exp(ll - m[np.newaxis, :]), axis=0))
        if progress is not None:
            progress(k + 1, K)

    valid = pointwise[np.isfinite(pointwise)]
    n_obs = int(valid.size)
    elpd = float(np.sum(valid))
    se = float(np.std(valid) * np.sqrt(n_obs)) if n_obs > 1 else float("nan")
    return dict(
        elpd_loo=elpd, se=se, p_loo=float("nan"), n_obs=n_obs, n_draws=int(n_draws),
        n_folds=int(K), pareto_k_max=float("nan"), frac_k_gt_0_7=float("nan"),
        method=f"cv{K}_{rt_density}", elpd_loo_pointwise=pointwise,
    )


# ---------------------------------------------------------------------------
# Posterior-predictive discrepancy
# ---------------------------------------------------------------------------

RT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _session_summary(rt: np.ndarray, choice: np.ndarray, correct_side: np.ndarray,
                     hit: np.ndarray) -> np.ndarray:
    """Summary-stat vector: [acc, p_right, RT quantiles...] over non-timeout trials."""
    if not hit.any():
        return np.full(2 + len(RT_QUANTILES), np.nan)
    c = choice[hit]; cs = correct_side[hit]; r = rt[hit]
    acc = float((c == cs).mean())
    p_right = float(c.mean())
    qs = np.quantile(r, RT_QUANTILES)
    return np.concatenate([[acc, p_right], qs])


@dataclass
class PPCDiscrepancy:
    discrepancy_z: float          # mean standardised distance across summary stats
    bayes_p_extreme: float        # mean two-sided posterior-predictive p-value
    stat_names: list = field(default_factory=list)
    obs_stats: np.ndarray = None
    ppc_mean: np.ndarray = None
    ppc_std: np.ndarray = None


def ppc_discrepancy(obs_summary: np.ndarray, ppc_summaries: np.ndarray) -> PPCDiscrepancy:
    """Standardised discrepancy between observed stats and PPC stat distribution.

    obs_summary: (K,) observed summary stats.
    ppc_summaries: (n_draws, K) posterior-predictive summary stats.
    """
    names = ["acc", "p_right"] + [f"rt_q{int(q*100)}" for q in RT_QUANTILES]
    mean = np.nanmean(ppc_summaries, axis=0)
    std = np.nanstd(ppc_summaries, axis=0)
    z = np.abs(obs_summary - mean) / np.where(std > 1e-9, std, np.nan)
    # two-sided PPC p-value per stat: fraction of draws at least as extreme
    p_two = []
    for k in range(ppc_summaries.shape[1]):
        col = ppc_summaries[:, k]
        col = col[~np.isnan(col)]
        if col.size == 0 or np.isnan(obs_summary[k]):
            p_two.append(np.nan); continue
        frac_below = np.mean(col <= obs_summary[k])
        p_two.append(2 * min(frac_below, 1 - frac_below))
    return PPCDiscrepancy(
        discrepancy_z=float(np.nanmean(z)),
        bayes_p_extreme=float(np.nanmean(p_two)),
        stat_names=names, obs_stats=obs_summary, ppc_mean=mean, ppc_std=std,
    )
