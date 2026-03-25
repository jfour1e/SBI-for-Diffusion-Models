from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..run_config import RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL
cfg = RUN_CONFIG_PARAMS

from .rt_choice_model import max_num_pulses, as_pulse_tensor, generate_pulses_torch, _ou_transition_params

def _run_fine_ou_loop_lapse(
    a0_frac: Tensor,
    lam: Tensor,
    v: Tensor,
    B: Tensor,
    tau: Tensor,
    s: Tensor,
    *,
    mu_sensory: float,
    dt_internal: float,
    pulse_interval: float,
    T_MAX: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Simulate non-lapse pulse-accumulation trials with leaky OU dynamics and absorbing bounds.
    Returns hit indicators, binary choices, and decision times before adding tau.
    """
    device = a0_frac.device
    dtype = a0_frac.dtype
    N = a0_frac.shape[0]

    delta = float(pulse_interval)
    dt0 = float(dt_internal)

    # force an integer number of internal steps per pulse interval
    steps_per_pulse = max(1, int(round(delta / dt0)))
    dt = delta / steps_per_pulse

    P_max = s.shape[1]

    # available decision time after non-decision delay tau
    max_dec_time = (float(T_MAX) - tau).clamp_min(0.0)
    step_limit = torch.floor(max_dec_time / dt).to(torch.int64)
    max_steps = int(step_limit.max().item()) if N > 0 else 0

    decay, noise_std = _ou_transition_params(lam, dt, float(mu_sensory))

    a = a0_frac * B
    hit = torch.zeros((N,), dtype=torch.bool, device=device)
    choice = torch.zeros((N,), dtype=torch.int64, device=device)
    decision_time = torch.zeros((N,), dtype=dtype, device=device)

    for step in range(max_steps + 1):
        active = (~hit) & (step < step_limit)
        if not torch.any(active):
            break

        # pulse kick at pulse boundaries — pulse k fires at t=(k+1)*delta,
        # matching base model convention and mask_unperceived_pulses.
        if step > 0 and step % steps_per_pulse == 0:
            k = step // steps_per_pulse - 1
            if k < P_max:
                a = torch.where(active, a + v * s[:, k], a)

                hit_upper = active & (a >= B)
                hit_lower = active & (a <= 0.0)
                newly_hit = hit_upper | hit_lower
                if torch.any(newly_hit):
                    hit[newly_hit] = True
                    choice[hit_upper] = 1
                    choice[hit_lower] = 0
                    decision_time[newly_hit] = step * dt

        active = (~hit) & (step < step_limit)
        if not torch.any(active):
            continue

        # OU evolution for one internal step (restores toward B/2, not 0)
        eps = torch.randn((N,), device=device, dtype=dtype)
        a = torch.where(active, a * decay + (B / 2.0) * (1.0 - decay) + noise_std * eps, a)

        hit_upper = active & (a >= B)
        hit_lower = active & (a <= 0.0)
        newly_hit = hit_upper | hit_lower
        if torch.any(newly_hit):
            hit[newly_hit] = True
            choice[hit_upper] = 1
            choice[hit_lower] = 0
            decision_time[newly_hit] = (step + 1) * dt

    return hit, choice, decision_time

def _sample_lapse_observations(
    n_trials: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    rt_max: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Sample lapse trials with uniform RT on (0, T_MAX-eps) and random choice with p=0.5."""
    if n_trials == 0:
        return (
            torch.empty((0,), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.int64),
        )

    eps = 1e-6
    upper = float(T_MAX if rt_max is None else rt_max)
    upper = max(eps, upper - eps)

    rt = torch.rand((n_trials,), device=device, generator=generator, dtype=dtype) * upper
    choice = torch.randint(
        low=0,
        high=2,
        size=(n_trials,),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return rt, choice


def simulate_rt_choice_batch_lapse(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: float = cfg.P_SUCCESS,
    pulse_generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Simulate a batch of pulse-based RT/choice trials with a lapse mode.

    theta columns:
      [a0_frac, lam, v, B, tau, p_lapse]

    Returns:
      x:   (N, 2) with columns [rt, choice]
      hit: (N,) bool; lapse trials are marked hit=True so they are never retried
      s:   (N, P_max) pulse sequences shown on each trial
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 6:
        raise ValueError(f"Expected theta shape (N,6) or (6,), got {tuple(theta.shape)}")

    device = theta.device
    dtype = torch.float32
    theta = theta.to(dtype=dtype)

    N = theta.shape[0]
    a0_frac = theta[:, 0].clamp(0.0, 1.0)
    lam = theta[:, 1].abs()
    v = theta[:, 2].abs()
    B = theta[:, 3].abs().clamp_min(1e-6)
    tau = theta[:, 4].clamp(0.0, float(T_MAX) - 1e-6)
    p_lapse = theta[:, 5].clamp(0.0, 1.0)

    P_max = max_num_pulses()
    delta = float(PULSE_INTERVAL)

    # pulse matrix in {-1, +1}
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

    dt_internal = float(getattr(cfg, "DT_INTERNAL", 0.01))

    # sample trialwise lapse indicators
    is_lapse = torch.rand(
        (N,), device=device, generator=pulse_generator, dtype=dtype
    ) < p_lapse

    rt = torch.empty((N,), device=device, dtype=dtype)
    choice = torch.empty((N,), device=device, dtype=torch.int64)
    hit = torch.empty((N,), device=device, dtype=torch.bool)

    # non-lapse trials: original accumulator dynamics
    non_lapse_idx = (~is_lapse).nonzero(as_tuple=False).squeeze(1)
    if non_lapse_idx.numel() > 0:
        idx = non_lapse_idx

        hit_nl, choice_nl, decision_time_nl = _run_fine_ou_loop_lapse(
            a0_frac=a0_frac[idx],
            lam=lam[idx],
            v=v[idx],
            B=B[idx],
            tau=tau[idx],
            s=s[idx],
            mu_sensory=float(mu_sensory),
            dt_internal=dt_internal,
            pulse_interval=delta,
            T_MAX=float(T_MAX),
        )

        rt_nl = (tau[idx] + decision_time_nl).clamp(1e-6, float(T_MAX))
        rt[idx] = rt_nl
        choice[idx] = choice_nl
        hit[idx] = hit_nl

    # lapse trials: random RT and random choice
    lapse_idx = is_lapse.nonzero(as_tuple=False).squeeze(1)
    if lapse_idx.numel() > 0:
        idx = lapse_idx

        rt_l, choice_l = _sample_lapse_observations(
            int(idx.numel()),
            device=device,
            dtype=dtype,
            generator=pulse_generator,
            rt_max=float(T_MAX),
        )

        rt[idx] = rt_l.clamp(1e-6, float(T_MAX))
        choice[idx] = choice_l
        hit[idx] = True

    x = torch.stack([rt, choice.to(dtype)], dim=-1)
    return x, hit, s