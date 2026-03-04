# SBI-for-Diffusion-Models

Simulation-based inference (SBI) for a **pulse-based drift–diffusion model** using session-level Neural Posterior Estimation (NPE). 

This repo is designed for **amortized inference**: train once on many simulated sessions, then sample posteriors quickly for new sessions **without MCMC**.

The project uses **[`uv`](https://github.com/astral-sh/uv)** for fast, reproducible Python environments (shout out Ryan for this one). 
---

## What’s implemented

### Simulator (PyTorch)
- Pulse-based accumulator model with:
  - OU noise between pulses
  - Pulse “kicks” at fixed intervals
  - Absorbing bounds (0 and `B`)
  - Non-decision time `tau`
- Generates per-trial outputs:
  - `rt` (reaction time)
  - `choice` (0/1)
  - `hit` mask (timeout vs. decision)
  - pulse sequence `s` (±1)

### Session formatting
- Each session contains `T` trials.
- Trials that timeout are retried

Final flattened representation:
- `x_session`: shape `(N_sessions, T * (2 + P + 1))`
  - `2` = `[rt, choice]`
  - `P` = pulses per trial

### NPE / MNPE model (sbi)
- `sbi` NPE with an NSF flow (`posterior_nn(model="nsf")`)
- Custom embedding: `MaskAwarePermutationInvariantEmbedding`
  - embeds each trial (excluding mask)
  - multiplies by mask
  - aggregates across trials (mean or sum)

## Requirements

- Python **>= 3.10**
- `uv` (package & environment manager)

---

## Installation

### Install UV on macOS 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment 
```bash 
uv venv .venv
source .venv/bin/activate
```

### Install Dependencies 
```bash
uv pip install -e . 
uv sync 
```

### Add dependencies 
```bash 
uv add numpy pandas matplotlib torch sbi
```

This section describes the full **session-level NPE (Neural Posterior Estimation)** workflow for the pulse-based RT–choice model.

Unlike MNLE, this pipeline **does NOT learn a likelihood** and **does NOT use MCMC**.

Instead, NPE learns an amortized posterior:
\[
p(\theta \mid x)
\]

which allows **direct posterior sampling** from observed session data without any iterative inference procedure.

The full workflow consists of:

1. Simulate session-level training data
2. Train neural posterior estimator (NPE)
3. Simulate an observed dataset
4. Perform direct posterior sampling
5. Perform Simulation-Based Calibration (SBC)

Run:

- **rt_choice_pipeline_mnpe.py**  
to train a neural posterior estimator from scratch and run inference

or

- **rt_choice_pipeline_mnpe_from_pretrained.py**  
to load a pretrained posterior model and run inference only.

---

## Configuration

All experiment parameters live in:

src/sbi_for_diffusion_models/run_config.py

Key MNLE controls:

| Parameter | Description |
|----------|-------------|
| `NPE_NUM_SESSIONS` | Number of simulated training sessions |
| `NUM_TRIALS_OBS` | Trials per observed dataset |
| `NPE_TRAIN_BATCH_SIZE` | Neural posterior training batch size |
| `NPE_HIDDEN_FEATURES` | Flow network hidden dimension |
| `NPE_NUM_TRANSFORMS` | Number of NSF flow transforms |
| `NPE_NUM_BINS` | Spline bins per transform |
| `NPE_EMBEDDING_OUTPUT_DIM` | Session embedding dimension |
| `NPE_POSTERIOR_SAMPLES` | Number of posterior samples drawn at inference |
| `RUN_SBC` | Enable Simulation-Based Calibration |
| `NPE_SBC_NUM_DATASETS` | Number of SBC datasets |
| `NPE_SBC_POST_SAMPLES` | Posterior samples per SBC dataset |

---

## Summary

The NPE pipeline consists of:

1. Simulating session-level training datasets from the pulse-based RT-choice model  
2. Embedding session data using a permutation-invariant DeepSets architecture  
3. Training a neural posterior estimator via Neural Posterior Estimation (NPE)  
4. Performing **direct posterior sampling** from observed session data  
5. Verifying posterior calibration using Simulation-Based Calibration (SBC)

This approach provides:

- Fully amortized posterior inference  
- Direct sampling from \( p(\theta \mid x) \)  
- Support for variable effective trial counts (timeouts handled internally)  
- Orders-of-magnitude faster inference than MCMC  
- SBC diagnostics for posterior calibration  
