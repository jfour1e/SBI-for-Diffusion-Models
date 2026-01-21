# SBI-for-Diffusion-Models

We implement simulation-based inference (SBI) for pulse-based Drift–Diffusion Models (DDMs) using neural likelihood estimation (MNLE) and Bayesian inference with MCMC. 

We use:

- PyTorch for simulation and neural networks
- **['sbi (v0.25.0)'](https://github.com/sbi-dev/sbi)** for neural likelihoods and MCMC
- **[`uv`](https://github.com/astral-sh/uv)** for virtual enviroment handling 
---

## Installing `uv`

### macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment 
```bash 
uv .venv
source venv/bin/activate
```

### Install Dependencies 
```bash
uv sync 
```
---

## Basic usage 

1. Simulate Training Dataset 
```python 

n_max, steps_per_pulse = pulse_schedule()
P = n_pulses_max_from_schedule(n_max, steps_per_pulse)

# define prior over Theta 
prior_theta = build_prior_theta()

# Define training proposals over Theta 
pulse_prop = PulseSequenceProposal(P=P, p_success=cfg.P_SUCCESS, seed=0,device="cpu")
proposal_z = ExtendedProposal(theta_prior=prior_theta, pulse_proposal=pulse_prop, device="cpu")

# Simulate Training data 
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

# Summarize trial data 
summarize_trials("train (sample)", x_train[torch.randperm(len(x_train))[:50_000]])
```

2. Train neural likelihood (MNLE)
```python 
density_estimator = train_mnle(cfg, proposal_z, z_train, x_train, device="cpu")

# Save trained neural network for later use, Can use keyword name='model_trained.pt' for custom name
save_model(density_estimator, cfg)

# Simulate Observed Session 
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
```

3. Inference ONLY, load saved model: 
```python 
# Load previously trained model. Can pass in name='model_name.pt' for custom model loading 
density_estimator = load_model(cfg, proposal_z, device="cpu")

# run Inference - can be done after training or in isolation 
samples = run_inference_mcmc(cfg, prior_theta, density_estimator, x_o, pulses_o)
```

4. Simulation-based Calibration (SBC)
To verify posterior correctness, run SBC: 
```python 
run_sbc(
    cfg,
    prior_theta=prior_theta,
    density_estimator=density_estimator,
    device="cpu",
    num_datasets=cfg.SBC_NUM_DATASETS,
    posterior_samples_per_dataset=cfg.SBC_POST_SAMPLES,
    seed=0,
    param_names=("a0", "lam", "v", "B", "tau"),
    outdir=sbc_outdir,
    plot_bins=30,
)
```
This performs repeated cycles of:

- Sample $\theta$ ~ prior
- Simulate dataset
- Run MCMC posterior
- Compute rank statistics
- Plot rank histograms

Uniform rank histograms indicate well-calibrated inference.

## Configuration 
All experiment parameters live in sbi_for_diffusion_models/run_config.py

Key controls include 
```bash 
NUM_SIMULATIONS      # MNLE training size
NUM_TRIALS_OBS       # Trials per dataset
POSTERIOR_SAMPLES    # MCMC samples
SBC_NUM_DATASETS     # Number of SBC repetitions
SBC_POST_SAMPLES     # MCMC samples per SBC dataset
```