from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..run_config import RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL
cfg = RUN_CONFIG_PARAMS

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

def mask_unperceived_pulses(
    pulse_sides: Tensor,
    rt: Tensor,
    pulse_interval: float = float(PULSE_INTERVAL),
) -> Tensor:
    """
    Set pulse entries to NaN for any pulse the animal could not have perceived.
    """
    N, P = pulse_sides.shape
    # Pulses arrive at t = (k+1)*pulse_interval (first pulse at pulse_interval, not 0)
    pulse_times = (torch.arange(P, device=pulse_sides.device, dtype=torch.float32) + 1) * pulse_interval
    unperceived = pulse_times.unsqueeze(0) > rt.to(pulse_sides.device).unsqueeze(1)  # (N, P)
    out = pulse_sides.clone()
    out[unperceived] = float("nan")
    return out


def pack_x_rt_choice(rt_choice: Tensor, *, log_rt: bool) -> Tensor:
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)

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

def _run_fine_ou_loop(
    a0_frac: Tensor,
    lam: Tensor,
    v: Tensor,
    B: Tensor,
    tau: Tensor,
    s: Tensor,                      # (N, P_max) in {-1,+1}
    *,
    mu_sensory: float,
    dt_internal: float,
    pulse_interval: float,
    T_MAX: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fine-step OU simulator:
      - internal OU step dt_internal (e.g. 10ms)
      - pulse kicks every pulse_interval (e.g. 250ms) at t = k*pulse_interval (k=0,1,...)
      - checks boundary after pulse kick and after OU step
    Returns:
      hit: (N,) bool
      choice: (N,) int64 0=lower, 1=upper (only meaningful when hit=True)
      decision_time: (N,) float32 (0..T_MAX-tau), only meaningful when hit=True
    """
    device = a0_frac.device
    dtype = a0_frac.dtype
    N = a0_frac.shape[0]

    delta = float(pulse_interval)
    dt0 = float(dt_internal)

    # Make dt divide the pulse interval exactly (avoids drift)
    steps_per_pulse = max(1, int(round(delta / dt0)))
    dt = delta / steps_per_pulse  # adjusted dt used in simulation

    # Max number of pulses implied by T_MAX/pulse_interval
    P_max = s.shape[1]

    # Continuous decision window length for each trial
    max_dec_time = (float(T_MAX) - tau).clamp_min(0.0)  # (N,)
    step_limit = torch.floor(max_dec_time / dt).to(torch.int64)  # (N,)
    max_steps = int(step_limit.max().item()) if N > 0 else 0

    # OU params per dt
    decay, noise_std = _ou_transition_params(lam, dt, float(mu_sensory))  # (N,), (N,)

    a = a0_frac * B
    hit = torch.zeros((N,), dtype=torch.bool, device=device)
    choice = torch.zeros((N,), dtype=torch.int64, device=device)
    decision_time = torch.zeros((N,), dtype=dtype, device=device)

    for step in range(max_steps + 1):
        active = (~hit) & (step < step_limit)
        if not torch.any(active):
            break

        if step > 0 and step % steps_per_pulse == 0:
            k = step // steps_per_pulse - 1  # pulse 0 at step=steps_per_pulse
            if k < P_max:
                # apply kick to active trials
                a = torch.where(active, a + v * s[:, k], a)

                # check boundary immediately after kick
                hit_upper = active & (a >= B)
                hit_lower = active & (a <= 0.0)
                newly_hit = hit_upper | hit_lower
                if torch.any(newly_hit):
                    hit[newly_hit] = True
                    choice[hit_upper] = 1
                    choice[hit_lower] = 0
                    decision_time[newly_hit] = step * dt  # exact pulse time

        # remaining active after pulse check
        active = (~hit) & (step < step_limit)
        if not torch.any(active):
            continue

        eps = torch.randn((N,), device=device, dtype=dtype)
        a = torch.where(active, a * decay + (B / 2.0) * (1.0 - decay) + noise_std * eps, a)

        # boundary check after OU step (at t=(step+1)*dt)
        hit_upper = active & (a >= B)
        hit_lower = active & (a <= 0.0)
        newly_hit = hit_upper | hit_lower
        if torch.any(newly_hit):
            hit[newly_hit] = True
            choice[hit_upper] = 1
            choice[hit_lower] = 0
            decision_time[newly_hit] = (step + 1) * dt

    return hit, choice, decision_time

def simulate_rt_choice_batch(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: float = cfg.P_SUCCESS,
    pulse_generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 5:
        raise ValueError(f"Expected theta shape (N,5) or (5,), got {tuple(theta.shape)}")

    device = theta.device
    dtype = torch.float32
    theta = theta.to(dtype=dtype)

    N = theta.shape[0]
    a0_frac = theta[:, 0].clamp(0.0, 1.0)
    lam = theta[:, 1].abs()  # keep nonnegative
    v = theta[:, 2].abs()
    B = theta[:, 3].abs().clamp_min(1e-6)
    tau = theta[:, 4].clamp(0.0, float(T_MAX) - 1e-6)

    delta = float(PULSE_INTERVAL)
    P_max = max_num_pulses()

    # pulses in {-1,+1}
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

    # internal step (add DT_INTERNAL to your config)
    dt_internal = float(getattr(cfg, "DT_INTERNAL", 0.01))

    hit, choice, decision_time = _run_fine_ou_loop(
        a0_frac=a0_frac,
        lam=lam,
        v=v,
        B=B,
        tau=tau,
        s=s,
        mu_sensory=float(mu_sensory),
        dt_internal=dt_internal,
        pulse_interval=delta,
        T_MAX=float(T_MAX),
    )

    rt = (tau + decision_time).clamp(1e-6, float(T_MAX))
    x = torch.stack([rt, choice.to(dtype)], dim=-1)
    return x, hit, s