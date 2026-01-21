from __future__ import annotations

import os
import time
import cProfile
import re
import pstats
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from torch.distributions import Beta, LogNormal, Distribution
from sbi.utils import MultipleIndependent

from sbi_for_diffusion_models.proposals import PulseSequenceProposal, ExtendedProposal
from sbi_for_diffusion_models.models.rt_choice_model import pulse_schedule, n_pulses_max_from_schedule
from sbi_for_diffusion_models.data_simulator import (
    simulate_observed_session,
    simulate_training_set_with_conditions,
    summarize_trials,
)
from sbi_for_diffusion_models.mnle import train_mnle, run_inference_mcmc, run_sbc, save_model, load_model
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS

cfg = RUN_CONFIG_PARAMS


def build_prior_theta() -> Distribution:
    """Prior over theta = [a0, lam, v, B, tau]."""
    return MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),          # a0
            LogNormal(torch.tensor([-1.0]), torch.tensor([1.0])),     # lam
            LogNormal(torch.tensor([0.0]), torch.tensor([1.0])),      # v
            LogNormal(torch.tensor([2.75]), torch.tensor([0.5])),     # B
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),           # tau
        ]
    )


def _dump_profile(prof: cProfile.Profile, outpath: str, top_n: int = 60) -> None:
    """Write .prof and a human-readable .txt next to it."""
    outpath = str(outpath)
    Path(os.path.dirname(outpath) or ".").mkdir(parents=True, exist_ok=True)
    prof.dump_stats(outpath)

    txt_path = outpath + ".txt"
    with open(txt_path, "w") as f:
        stats = pstats.Stats(prof, stream=f)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(top_n)
    print(f"[profiler] wrote: {outpath}")
    print(f"[profiler] wrote: {txt_path}")


def profile_block(label: str, fn, *, outdir: str, top_n: int = 60):
    """Profile a callable and return its result."""
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    out = fn()
    prof.disable()
    dt = time.time() - t0

    outpath = os.path.join(outdir, f"{label}.prof")
    _dump_profile(prof, outpath, top_n=top_n)
    print(f"[profiler] {label} wall time: {dt:.2f}s")
    return out


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- setup ----
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    prior_theta = build_prior_theta()

    pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0, device="cpu")
    proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

    outdir = os.environ.get("OUTDIR", "profile_outputs")
    os.makedirs(outdir, exist_ok=True)

    # ---- train or load model ----
    # Recommendation: for profiling MCMC/SBC, load if available to avoid retraining every run.
    density_estimator = load_model(cfg, proposal_z, device="cpu")

    if density_estimator is None:
        print("\n--- Simulating training set ---")
        z_train, x_train = simulate_training_set_with_conditions(
            proposal=proposal_z,
            num_simulations=cfg.NUM_SIMULATIONS,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            device="cpu",
            mu_sensory=cfg.MU_SENSORY,
            p_success=cfg.P_SUCCESS,
            P=P,
            log_rt=cfg.LOG_RT_MANUALLY,
        )
        summarize_trials("train (sample)", x_train[torch.randperm(len(x_train))[:50_000]])

        print("\n--- Training MNLE ---")
        density_estimator = train_mnle(cfg, proposal_z, z_train, x_train, device="cpu")
        save_model(density_estimator, cfg)
    else:
        print("[info] Loaded MNLE model from ~/models. Skipping training.")

    # ---- simulate observed dataset once ----
    if cfg.THETA_TRUE_FROM_PRIOR:
        theta_true = prior_theta.sample((1,)).view(5)
    else:
        raise ValueError("Set THETA_TRUE_FROM_PRIOR=True or provide your own theta_true.")

    x_o, pulses_o = simulate_observed_session(
        theta_true,
        num_trials=cfg.NUM_TRIALS_OBS,
        device="cpu",
        mu_sensory=cfg.MU_SENSORY,
        p_success=cfg.P_SUCCESS,
        P=P,
        seed=123,
        log_rt=cfg.LOG_RT_MANUALLY,
    )
    summarize_trials("observed", x_o)
    print("theta_true:", theta_true.detach().cpu().numpy().round(4).tolist())

    # ---- profile MCMC (single dataset) ----
    def _mcmc_call():
        return run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o)

    samples = profile_block("mcmc_single_dataset", _mcmc_call, outdir=outdir, top_n=80)
    np.save(os.path.join(outdir, "posterior_samples_theta.npy"), samples.numpy())
    print("Saved:", os.path.join(outdir, "posterior_samples_theta.npy"))

    # ---- profile SBC (multiple datasets) ----
    # SBC can be insanely expensive if cfg.SBC_NUM_DATASETS * cfg.SBC_POST_SAMPLES is large.
    # For profiling, consider overriding them here without touching global cfg:
    cfg_sbc = cfg
    # Example overrides (uncomment to reduce for profiling)
    # cfg_sbc = replace(cfg, SBC_NUM_DATASETS=10, SBC_POST_SAMPLES=500)

    sbc_outdir = os.path.join(outdir, "sbc")
    os.makedirs(sbc_outdir, exist_ok=True)

    def _sbc_call():
        return run_sbc(
            cfg_sbc,
            prior_theta=prior_theta,
            density_estimator=density_estimator,
            device="cpu",
            num_datasets=cfg_sbc.SBC_NUM_DATASETS,
            posterior_samples_per_dataset=cfg_sbc.SBC_POST_SAMPLES,
            seed=0,
            param_names=("a0", "lam", "v", "B", "tau"),
            outdir=sbc_outdir,
            plot_bins=30,
        )

    profile_block("sbc", _sbc_call, outdir=outdir, top_n=80)

    print("\nDone. Profiles are in:", outdir)
    print("Tip: open .prof with snakeviz:")
    print(f"  uv run python -m pip install snakeviz")
    print(f"  uv run snakeviz {os.path.join(outdir, 'mcmc_single_dataset.prof')}")


if __name__ == "__main__":
    main()
