from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

# Global simulation constants
DT = 0.005            # time step 
T_MAX = 8.0           # total RT window 
T0 = 0.1              # fixed non-decision time 
T_DEC = T_MAX - T0    # diffusion window 

N_STEPS = int(T_DEC / DT)

PULSE_INTERVAL = 0.1
STEPS_PER_PULSE = int(PULSE_INTERVAL / DT)
STEPS_PER_PULSE = max(STEPS_PER_PULSE, 1)
N_PULSES = N_STEPS // STEPS_PER_PULSE + 1


"""
    Define logic for single evolution of DDM. Includes 
    - bound height logic
    - parameter constraints
    - pulse schedule
    - diffusion loop 

    Theta = [bias, lam, nu, B, sigma_a, sigma_s]

    bias      : starting point as fraction of bound [0,1]
    lam       : leak rate (1/s)
    nu        : pulse magnitude
    B         : bound height
    sigma_a   : accumulator noise
    sigma_s   : sensory noise SD (eta ~ N(1, sigma_s^2))
"""
def simulate_pulse_ddm_single(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_sensory: float = 1.0,
) -> tuple[float, int]:
    """
    Returns:
        rt (float), choice_idx (0 = lower/left, 1 = upper/right)
    """
   
    a0, lam, nu, B, sigma_a, t_nd, sigma_s = theta.astype(np.float32)

    # enforce basic constraints
    B = float(abs(B))
    lam = float(abs(lam))
    nu = float(abs(nu))
    sigma_a = float(abs(sigma_a))
    sigma_s = float(abs(sigma_s))

    # starting point in [0, B]
    a0 = float(np.clip(a0, 0.0, 1.0))
    a_curr = a0 * B

    # per-trial nondecision time: allow override but keep it sane
    if not np.isfinite(t_nd):
        t_nd = T0
    t_nd = float(np.clip(t_nd, 0.0, T_MAX - 1e-3))

    # diffusion window depends on t_nd
    t_dec = T_MAX - t_nd
    n_steps = int(t_dec / DT)
    if n_steps <= 1:
        # degenerate: almost no decision time, return random-ish
        choice_idx = 1 if a_curr >= (B / 2.0) else 0
        rt = float(np.clip(t_nd, 1e-3, T_MAX))
        return rt, int(choice_idx)

    # pulses: equally spaced by STEPS_PER_PULSE
    steps_per_pulse = STEPS_PER_PULSE
    n_pulses = n_steps // steps_per_pulse + 1

    # pulse sides: R=+1, L=-1 (random here; later you can feed in a stimulus sequence)
    s_seq = np.where(rng.random(size=n_pulses) < 0.5, 1.0, -1.0).astype(np.float32)

    # pulse times (indices)
    pulse_indices = np.arange(0, n_steps, steps_per_pulse, dtype=int)
    pulse_indices = pulse_indices[pulse_indices < n_steps]
    s_seq = s_seq[: pulse_indices.shape[0]]

    done = False
    first_hit_time = -1.0
    choice_idx = 0

    sqrt_dt = np.sqrt(DT)

    for t in range(n_steps):
        # leak + diffusion
        drift = (-lam * a_curr) * DT
        noise = sigma_a * sqrt_dt * rng.normal()

        a_next = a_curr + drift + noise

        # if pulse happens now, add jump s * nu * eta
        # (eta ~ N(1, sigma_s^2); if sigma_s=0 => eta=1 exactly)
        if t in set(pulse_indices):  # simple & clear; can micro-opt later
            k = int(np.where(pulse_indices == t)[0][0])  # index of this pulse
            if sigma_s > 0.0:
                eta = 1.0 + sigma_s * rng.normal()
            else:
                eta = 1.0

            jump = float(s_seq[k]) * nu * float(eta)
            a_next = a_next + jump

        # bounds: lower=0, upper=B
        if a_next >= B:
            done = True
            choice_idx = 1
            first_hit_time = (t + 1) * DT
            break
        elif a_next <= 0.0:
            done = True
            choice_idx = 0
            first_hit_time = (t + 1) * DT
            break

        a_curr = a_next

    if not done:
        first_hit_time = t_dec
        choice_idx = 1 if a_curr >= (B / 2.0) else 0

    rt = float(np.clip(t_nd + first_hit_time, 1e-3, T_MAX))
    return rt, int(choice_idx)

# variant of above function with no sensory noise 
def simulate_pulse_ddm_single_no_sensory_noise(
    theta: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, int]:
    theta2 = theta.copy().astype(np.float32)
    theta2[5] = 0.0  # sigma_s
    return simulate_pulse_ddm_single(theta2, rng, mu_sensory=1.0)


# torch based simulator for runtime efficiency 
def pulse_ddm_simulator_torch(theta: Tensor, rng: np.random.Generator | None = None, **sim_kwargs) -> Tensor:
    """
    theta: (batch, D)
    returns: (batch, 2) with columns [rt, choice]
    """
    theta_np = theta.detach().cpu().numpy().astype(np.float32)
    batch_size = theta_np.shape[0]
    xs = np.zeros((batch_size, 2), dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    for i in range(batch_size):
        rt, choice_idx = simulate_pulse_ddm_single(theta_np[i], rng, **sim_kwargs)
        xs[i, 0] = rt
        xs[i, 1] = choice_idx

    return torch.from_numpy(xs).to(torch.float32)


def simulate_session_data(theta_true: Tensor, num_trials: int = 1000, rng: np.random.Generator | None = None, **sim_kwargs) -> Tensor:
    theta_np = theta_true.detach().cpu().numpy().astype(np.float32)
    xs = np.zeros((num_trials, 2), dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    for n in range(num_trials):
        rt, choice_idx = simulate_pulse_ddm_single(theta_np, rng, **sim_kwargs)
        xs[n, 0] = rt
        xs[n, 1] = choice_idx

    return torch.from_numpy(xs).to(torch.float32)