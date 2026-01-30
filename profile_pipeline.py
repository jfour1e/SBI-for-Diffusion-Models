# scripts/profile_pipeline.py
from __future__ import annotations

import os
import argparse
import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot
from sbi.utils import MultipleIndependent
from torch.distributions import Beta, LogNormal, Distribution

from sbi_for_diffusion_models.proposals import PulseSequenceProposal, ExtendedProposal
from sbi_for_diffusion_models.models.rt_choice_model import pulse_schedule, n_pulses_max_from_schedule
from sbi_for_diffusion_models.data_simulator import (
    simulate_observed_session,
    simulate_training_set_with_conditions,
    summarize_trials,
)
from sbi_for_diffusion_models.mnle import train_mnle, run_inference_mcmc, run_sbc, save_model, load_model
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS
from sbi_for_diffusion_models.profile_utils import profile_block, tracemalloc_top, profile_with_cprofile
import tracemalloc

cfg = RUN_CONFIG_PARAMS

def build_prior_theta() -> Distribution:
    return MultipleIndependent(
        [
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),           # a0
            LogNormal(torch.tensor([-1.0]), torch.tensor([1.0])),     # lam
            LogNormal(torch.tensor([0.0]), torch.tensor([1.0])),      # v
            LogNormal(torch.tensor([2.75]), torch.tensor([0.5])),     # B
            Beta(torch.tensor([2.0]), torch.tensor([2.0])),           # tau (placeholder)
        ]
    )


def _build_proposals(P: int):
    prior_theta = build_prior_theta()
    pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0, device="cpu")
    proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")
    return prior_theta, proposal_z


def _maybe_get_model(prior_theta, proposal_z, *, device: str, train_if_missing: bool):
    # Try load first
    density_estimator = load_model(cfg, proposal_z, device=device)
    if density_estimator is not None:
        return density_estimator

    if not train_if_missing:
        raise RuntimeError("No saved MNLE model found, and train_if_missing=False.")

    # Train (this can be expensive; you can profile it too if you want)
    print("\n--- Simulating training set ---")
    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

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

    print("\n--- Training MNLE ---")
    density_estimator = train_mnle(cfg, proposal_z, z_train, x_train, device=device)
    save_model(density_estimator, cfg)
    return density_estimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["mcmc", "sbc", "all"], default="mcmc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--profile-out", default="pipeline_profile.pstats")
    parser.add_argument("--sbc-datasets", type=int, default=None, help="Override cfg.SBC_NUM_DATASETS for profiling.")
    parser.add_argument("--post-samples", type=int, default=None, help="Override cfg.SBC_POST_SAMPLES for profiling.")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    n_max, steps_per_pulse = pulse_schedule()
    P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
    print("P =", P, "pulses per trial")

    prior_theta, proposal_z = _build_proposals(P)
    density_estimator = _maybe_get_model(prior_theta, proposal_z, device=args.device, train_if_missing=args.train_if_missing)

    # Simulate ONE observed dataset (used for MCMC profiling stage)
    theta_true = prior_theta.sample((1,)).view(5)
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

    # ---- profiling targets ----
    def run_mcmc_once():
        _ = run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o)

    def run_sbc_loop():
        outdir = os.path.join(os.environ.get("OUTDIR", "mnle_outputs"), "sbc_profile")
        num_datasets = args.sbc_datasets if args.sbc_datasets is not None else cfg.SBC_NUM_DATASETS
        post_samples = args.post_samples if args.post_samples is not None else cfg.SBC_POST_SAMPLES

        run_sbc(
            cfg,
            prior_theta=prior_theta,
            density_estimator=density_estimator,
            device=args.device,
            num_datasets=num_datasets,
            posterior_samples_per_dataset=post_samples,
            seed=0,
            outdir=outdir,
        )

    # --- tracemalloc snapshot diff around the stage(s) ---
    tracemalloc.start(25)
    snap0 = tracemalloc.take_snapshot()

    if args.stage in ("mcmc", "all"):
        with profile_block("MCMC (single dataset)"):
            # cProfile gives function-level timing; run it around the heavy call
            profile_with_cprofile(run_mcmc_once, outpath=args.profile_out, lines=60)

    if args.stage in ("sbc", "all"):
        with profile_block("SBC loop"):
            profile_with_cprofile(run_sbc_loop, outpath="sbc_" + args.profile_out, lines=60)

    snap1 = tracemalloc.take_snapshot()
    print("\n==== Tracemalloc allocation diff (after - before) ====")
    top_diff = snap1.compare_to(snap0, "lineno")
    for stat in top_diff[:25]:
        print(stat)

    tracemalloc.stop()


if __name__ == "__main__":
    main()
