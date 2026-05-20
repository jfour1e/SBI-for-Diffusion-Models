# SBI-for-Diffusion-Models

Simulation-based inference (SBI) for a pulse-based drift–diffusion model using
session-level Neural Posterior Estimation (NPE).

Train once on many simulated sessions, then sample posteriors quickly for new
sessions without MCMC. The project uses [`uv`](https://github.com/astral-sh/uv)
for environment management.

## Model

Pulse-driven OU accumulator with absorbing bounds. Two variants:

- `models/rt_choice_model.py` — base model with parameters
  `[a0, lam, v, B, tau]`.
- `models/lapse_rt_choice_model.py` — adds a lapse rate `p_lapse`; used as the
  default in pipelines.

Each session is `T` trials of `(rt, choice, pulse_1, ..., pulse_P)` flattened
to a single vector of length `T * (2 + P)` and used as the NPE conditioning
vector.

## Installation

Requires Python ≥ 3.11 and `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv sync
```

## Pipeline

1. **Train** an amortized NPE on simulated sessions mixed across
   `P_SUCCESS_TRAIN_VALUES`:
   ```bash
   python train_mixed_npe.py
   ```
2. **Fit marmoset data** at every (animal, stage), with optional posterior
   predictive checks:
   ```bash
   python scripts/fit_all_marmosets_all_stages.py
   ```
3. **Diagnostics**:
   - `scripts/prior_predictive_check.py` — verify the prior produces plausible
     RT / accuracy ranges before training.
   - `scripts/posterior_update_check.py` — measure per-parameter posterior
     contraction on simulated data.
   - `scripts/rt_choice_pipeline_sbc.py` — Simulation-Based Calibration rank
     histograms.

## Configuration

All hyperparameters live in
[`src/sbi_for_diffusion_models/run_config.py`](src/sbi_for_diffusion_models/run_config.py).
Key knobs:

| Parameter | Description |
|----------|-------------|
| `NUM_TRIALS_OBS` | Trials per observed dataset |
| `P_SUCCESS` | Single-condition stimulus reliability for diagnostic / inference simulations |
| `P_SUCCESS_TRAIN_VALUES` | Reliability values sampled across sessions when training one mixed NPE |
| `NPE_SESSIONS_PER_STEP` | Sessions simulated per training step |
| `NPE_NUM_STEPS` | Total training steps |
| `NPE_HIDDEN_FEATURES` | Flow network hidden dimension |
| `NPE_NUM_TRANSFORMS` | NSF coupling layers |
| `NPE_NUM_BINS` | Spline bins per transform |
| `NPE_EMBEDDING_OUTPUT_DIM` | Session embedding dimension |
| `NPE_POSTERIOR_SAMPLES` | Posterior samples drawn at inference |
| `RUN_SBC` | Enable Simulation-Based Calibration |
| `NPE_SBC_NUM_DATASETS` | Number of SBC datasets |
| `NPE_SBC_POST_SAMPLES` | Posterior samples per SBC dataset |
