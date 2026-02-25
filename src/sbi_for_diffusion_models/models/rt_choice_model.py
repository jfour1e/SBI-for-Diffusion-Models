from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..run_config import RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL,_DEFAULT_N_REFINE
cfg = RUN_CONFIG_PARAMS

# ---------------- Helper functions ----------------
def max_num_pulses() -> int:
    """Maximum number of pulses in a trial of duration T_MAX."""
    return int(float(T_MAX) / float(PULSE_INTERVAL))

def as_pulse_tensor(
    pulse_sides: Union[np.ndarray, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert pulse_sides to a (N,P) torch.Tensor on the desired device."""
    if isinstance(pulse_sides, Tensor):
        s = pulse_sides
    else:
        s = torch.from_numpy(np.asarray(pulse_sides))
    if s.ndim == 1:
        s = s.view(1, -1)
    if s.ndim != 2:
        raise ValueError(f"pulse_sides must have shape (N,P) or (P,), got {tuple(s.shape)}")
    return s.to(device=device, dtype=dtype)

def generate_pulses_torch(
    n_trials: int,
    n_pulses: int,
    *,
    p_success: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Torch-native pulse generator, values in {+1,-1}, shape (n_trials, n_pulses).
    """
    if n_trials < 0 or n_pulses < 0:
        raise ValueError("n_trials and n_pulses must be >= 0")
    if n_trials == 0 or n_pulses == 0:
        return torch.empty((n_trials, n_pulses), device=device, dtype=dtype)

    # correct_side in {+1,-1}
    # randint avoids making ones_like then where
    correct_side = torch.randint(
        low=0, high=2, size=(n_trials,), device=device, generator=generator
    )
    # map {0,1} -> {+1,-1}
    correct_side = (1 - 2 * correct_side).to(dtype)  # 0->+1, 1->-1

    is_correct = torch.rand((n_trials, n_pulses), device=device, generator=generator) < float(p_success)
    s = torch.where(is_correct, correct_side[:, None], -correct_side[:, None])
    return s.to(dtype)

# not needed anymore unless user wants to generate Numpy based matrix 
def generate_pulse_matrix_numpy(
    rng: np.random.Generator,
    n_trials: int,
    n_pulses: int,
    *,
    p_success: float = cfg.P_SUCCESS,
) -> np.ndarray:
    """
    Optional legacy NumPy pulse generator. Keep if you want NumPy-seeded stimuli.
    """
    if n_trials < 0:
        raise ValueError("n_trials must be >= 0")
    if n_pulses < 0:
        raise ValueError("n_pulses must be >= 0")
    if n_trials == 0 or n_pulses == 0:
        return np.empty((n_trials, n_pulses), dtype=np.float32)

    correct_side = np.where(rng.random(n_trials) < 0.5, 1.0, -1.0)
    is_correct = rng.random((n_trials, n_pulses)) < p_success
    return np.where(is_correct, correct_side[:, None], -correct_side[:, None]).astype(np.float32)


# ---------------- OU transition and coarse loop ----------------
def _ou_transition_params(
    lam: Tensor, dt: float, sigma: float,
) -> Tuple[Tensor, Tensor]:
    """Exact OU decay and noise-std for a step of size *dt*."""
    decay = torch.exp(-lam * dt)
    two_lam_dt = 2.0 * lam * dt
    var_factor = torch.where(
        lam.abs() < 1e-8,
        torch.full_like(lam, dt),
        (1.0 - torch.exp(-two_lam_dt)) / (2.0 * lam),
    )
    noise_std = float(sigma) * torch.sqrt(var_factor.clamp_min(1e-30))
    return decay, noise_std

def _run_coarse_ou_loop(
    a0_frac: Tensor,
    v: Tensor,
    B: Tensor,
    decay: Tensor,
    noise_std: Tensor,
    s: Tensor,
    n_intervals: Tensor,
    P_max: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Coarse pass: one exact OU step per pulse interval (~80 iterations).

    Returns
    -------
    hit : (N,) bool
    choice : (N,) int64, 0=lower 1=upper
    hit_interval : (N,) int64, 1-based interval index of first crossing
    a_before_hit : (N,) float, accumulator at START of crossing interval
    hit_from_pulse : (N,) bool, True when the pulse kick caused the crossing
    """
    device = a0_frac.device
    dtype = a0_frac.dtype
    N = a0_frac.shape[0]

    a = a0_frac * B
    hit = torch.zeros((N,), dtype=torch.bool, device=device)
    choice = torch.zeros((N,), dtype=torch.int64, device=device)
    hit_interval = torch.zeros((N,), dtype=torch.int64, device=device)
    a_before_hit = torch.zeros((N,), dtype=dtype, device=device)
    hit_from_pulse = torch.zeros((N,), dtype=torch.bool, device=device)

    all_noise = torch.randn((P_max, N), device=device, dtype=dtype)

    for k in range(P_max):
        active = (~hit) & (k < n_intervals)
        if not torch.any(active):
            break

        # Save accumulator at start of interval only for potentially newly-hit trials.
        a_start = a

        # OU step over full pulse interval
        a = torch.where(active, a * decay + noise_std * all_noise[k], a)

        # Check boundaries after OU step
        hit_upper = active & (a >= B)
        hit_lower = active & (a <= 0.0)
        newly_hit = hit_upper | hit_lower
        if torch.any(newly_hit):
            hit_interval[newly_hit] = k + 1
            choice[hit_upper] = 1
            choice[hit_lower] = 0
            a_before_hit[newly_hit] = a_start[newly_hit]
            hit_from_pulse[newly_hit] = False
            hit[newly_hit] = True

        # Pulse kick for still active trials
        still_active = (~hit) & (k < n_intervals)
        if torch.any(still_active):
            a = torch.where(still_active, a + v * s[:, k], a)

            hit_upper = still_active & (a >= B)
            hit_lower = still_active & (a <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_interval[newly_hit] = k + 1
                choice[hit_upper] = 1
                choice[hit_lower] = 0
                a_before_hit[newly_hit] = a_start[newly_hit]
                hit_from_pulse[newly_hit] = True
                hit[newly_hit] = True

    return hit, choice, hit_interval, a_before_hit, hit_from_pulse


# -------------- MNPE Simulator --------------
def simulate_rt_choice_batch(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: float = cfg.P_SUCCESS,
    n_refine: int = _DEFAULT_N_REFINE,
    pulse_generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    MNPE-first simulator.

    Inputs:
      theta: (N,5) or (5,)
      pulse_sides:
        - If provided: (N,P_max) or (P_max,) or (1,P_max)
        - If None: generate pulses in torch on theta.device

    Outputs:
      x   : (N,2) float32 [rt, choice] (choice valid only where hit=True)
      hit : (N,) bool      True if decision occurred before timeout window
      s   : (N,P_max) float32 pulse sides actually used (so caller can store/flatten)
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 5:
        raise ValueError(f"Expected theta shape (N,5) or (5,), got {tuple(theta.shape)}")

    device = theta.device
    dtype = torch.float32
    theta = theta.to(dtype=dtype)

    N = theta.shape[0]
    a0_frac = theta[:, 0].clamp(0.0, 1.0)
    lam = theta[:, 1]
    v = theta[:, 2].abs()
    B = theta[:, 3].abs().clamp_min(1e-6)
    t_nd = theta[:, 4].clamp(0.0, float(T_MAX) - 1e-6)

    delta = float(PULSE_INTERVAL)
    sigma = float(mu_sensory)
    P_max = max_num_pulses()

    # decision window in full pulse intervals
    n_intervals = torch.floor((float(T_MAX) - t_nd) / delta).to(torch.int64).clamp(0, P_max)

    # coarse OU params
    decay_coarse, noise_std_coarse = _ou_transition_params(lam, delta, sigma)

    # pulse sides
    if pulse_sides is None:
        s = generate_pulses_torch(
            n_trials=N,
            n_pulses=P_max,
            p_success=float(p_success),
            device=device,
            dtype=dtype,
            generator=pulse_generator,
        )
    else:
        s = as_pulse_tensor(pulse_sides, device=device, dtype=dtype)
        if s.shape[0] == 1 and N > 1:
            s = s.expand(N, -1)
        if s.shape[0] != N:
            raise ValueError(f"pulse_sides first dim must match N={N} (or be 1), got {s.shape[0]}")
        if s.shape[1] < P_max:
            raise ValueError(f"pulse_sides has P={s.shape[1]} but needs at least {P_max} for T_MAX.")
        s = s[:, :P_max]

    # coarse pass
    hit, choice, hit_interval, a_before_hit, hit_from_pulse = _run_coarse_ou_loop(
        a0_frac=a0_frac,
        v=v,
        B=B,
        decay=decay_coarse,
        noise_std=noise_std_coarse,
        s=s,
        n_intervals=n_intervals,
        P_max=P_max,
    )

    # decision time (only meaningful where hit=True)
    decision_time = hit_interval.to(dtype) * delta

    # refine OU-step crossings
    ou_mask = hit & (~hit_from_pulse)
    if torch.any(ou_mask) and n_refine > 1:
        idx = torch.where(ou_mask)[0]
        M = idx.shape[0]
        k_cross = hit_interval[idx] - 1  # 0-based

        pulse_val = v[idx] * s[idx, k_cross]

        sub_delta = delta / n_refine
        decay_fine, noise_std_fine = _ou_transition_params(lam[idx], sub_delta, sigma)

        a_r = a_before_hit[idx].clone()
        hit_r = torch.zeros((M,), dtype=torch.bool, device=device)
        choice_r = torch.zeros((M,), dtype=torch.int64, device=device)
        hit_substep = torch.full((M,), n_refine, dtype=torch.int64, device=device)

        # Refine noise
        for t in range(n_refine):
            active_r = ~hit_r
            if not torch.any(active_r):
                break
            eps = torch.randn((M,), device=device, dtype=dtype)
            a_r = torch.where(active_r, a_r * decay_fine + noise_std_fine * eps, a_r)

            hit_upper = active_r & (a_r >= B[idx])
            hit_lower = active_r & (a_r <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_substep[newly_hit] = t + 1
                choice_r[hit_upper] = 1
                choice_r[hit_lower] = 0
                hit_r[newly_hit] = True

        still_active = ~hit_r
        if torch.any(still_active):
            a_r = torch.where(still_active, a_r + pulse_val, a_r)
            hit_upper = still_active & (a_r >= B[idx])
            hit_lower = still_active & (a_r <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_substep[newly_hit] = n_refine
                choice_r[hit_upper] = 1
                choice_r[hit_lower] = 0
                hit_r[newly_hit] = True

        decision_time[idx] = k_cross.to(dtype) * delta + hit_substep.to(dtype) * sub_delta
        # update choice for refined hits
        choice[idx[hit_r]] = choice_r[hit_r]

    rt = (t_nd + decision_time).clamp(1e-6, float(T_MAX))
    x = torch.stack([rt, choice.to(dtype)], dim=-1)

    return x, hit, s

def pack_x_rt_choice(rt_choice: Tensor, *, log_rt: bool) -> Tensor:
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)