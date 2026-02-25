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

### Session formatting (mask-aware)
- Each session contains `T` trials.
- Trials that timeout are **dropped**, then the session is **padded back to length `T`** with:
  - zero-filled trial data
  - a mask bit `mask ∈ {0,1}` (1 = real, 0 = padded)

Final flattened representation:
- `x_session`: shape `(N_sessions, T * (2 + P + 1))`
  - `2` = `[rt, choice]`
  - `P` = pulses per trial
  - `1` = mask channel

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

## Quickstart 

This section describes the full trial-level MNLE (Mixed Neural Likelihood Estimation) workflow for the pulse-based RT-choice model.

MNLE learns an amortized likelihood:
\[
p(x \mid \theta)
\]
which can then be combined with a prior and used for posterior inference via MCMC.

The full workflow consists of:

1. Simulate training data
2. Train neural likelihood (MNLE)
3. Simulate an observed dataset
4. Run MCMC posterior inference
5. Perform Simulation-Based Calibration (SBC)

Run the **rt_choice_pipeline_mnpe.py** in order to train a neural network from scratch, or load a pretrained model and run **rt_choice_pipeline_mnpe_from_pretrained.py**. 

## Configuration

All experiment parameters live in:

src/sbi_for_diffusion_models/run_config.py

Key MNLE controls:

| Parameter | Description |
|----------|-------------|
| `NUM_SIMULATIONS` | MNLE training size |
| `TRAIN_BATCH_SIZE` | Simulator batch size |
| `NUM_TRIALS_OBS` | Trials per observed dataset |
| `POSTERIOR_SAMPLES` | Number of MCMC posterior samples |
| `SBC_NUM_DATASETS` | Number of SBC repetitions |
| `SBC_POST_SAMPLES` | MCMC samples per SBC dataset |
| `MCMC_METHOD` | Slice / HMC sampler |
| `NUM_CHAINS` | Parallel MCMC chains |
| `WARMUP_STEPS` | MCMC warmup steps |
| `TEMPERATURE` | Optional likelihood tempering (debug only) |

## Summary

The MNLE pipeline consists of:

1. Simulating trial-level training data from the pulse-based RT-choice model  
2. Training a neural likelihood estimator via Mixed Neural Likelihood Estimation (MNLE)  
3. Simulating an observed dataset  
4. Performing posterior inference using MCMC  
5. Verifying posterior calibration with Simulation-Based Calibration (SBC)

This approach provides:

- Amortized likelihood estimation  
- Full Bayesian posterior inference via MCMC  
- Support for mixed discrete-continuous observations (RT + choice)  
- SBC diagnostics for calibration  

MNLE is recommended when:

- Posterior fidelity is critical  
- MCMC runtime is acceptable  
- Likelihood-based diagnostics (e.g., SBC) are required  