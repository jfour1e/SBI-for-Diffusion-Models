from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .constants import T_MAX, PULSE_INTERVAL, DT_CHOICE
from .defaults import DEFAULT_P_SUCCESS
from .choice_model import generate_pulse_sides  # reuse your exact stimulus logic


@dataclass(frozen=True)
class RTChoiceModelParams:
    a0_frac: float
    lam: float
    v: float
    B: float
    t_nd: float

    @staticmethod
    def from_theta(theta: np.ndarray) -> "RTChoiceModelParams":
        if theta.shape[-1] != 5:
            raise ValueError(
                f"Expected theta with 5 params [a0, lam, v, B, t_nd], got shape {theta.shape}."
            )

        a0, lam, v, B, t_nd = theta.astype(np.float64)

        B = float(abs(B)) if np.isfinite(B) else 1.0
        B = max(B, 1e-6)

        a0 = float(np.clip(a0, 0.0, 1.0)) if np.isfinite(a0) else 0.5
        lam = float(lam) if np.isfinite(lam) else 0.0
        v = float(v) if np.isfinite(v) else 0.0

        t_nd = float(t_nd) if np.isfinite(t_nd) else 0.0
        t_nd = float(np.clip(t_nd, 0.0, float(T_MAX) - 1e-6))

        return RTChoiceModelParams(a0_frac=a0, lam=lam, v=v, B=B, t_nd=t_nd)


def _simulate_rt_choice_batch_torch(
    theta: Tensor,
    *,
    mu_sensory: float,
    p_success: float,
    rng: Optional[np.random.Generator],
) -> Tensor:
    """
    theta: (N,5) torch tensor on CPU or GPU
    returns: (N,2) float32 tensor: [rt, choice] where choice in {0,1}
    """
    device = theta.device
    dtype = torch.float32
    theta = theta.to(dtype=dtype)

    N = theta.shape[0]

    a0_frac = theta[:, 0].clamp(0.0, 1.0)
    lam = theta[:, 1]
    v = theta[:, 2].abs()
    B = theta[:, 3].abs().clamp_min(1e-6)
    t_nd = theta[:, 4].clamp(0.0, float(T_MAX) - 1e-6)

    dt = float(DT_CHOICE)
    n_max = int(np.floor(float(T_MAX) / dt))
    steps_per_pulse = max(int(np.round(float(PULSE_INTERVAL) / dt)), 1)

    # Decision window per trial in steps
    n_steps = torch.floor((float(T_MAX) - t_nd) / dt).to(torch.int64).clamp(0, n_max)

    # Start point in [0,B]
    a = a0_frac * B

    sigma = float(mu_sensory)
    sqrt_dt = np.sqrt(dt)

    # Outcomes
    hit = torch.zeros((N,), dtype=torch.bool, device=device)
    choice = torch.zeros((N,), dtype=torch.int64, device=device)
    hit_step = torch.zeros((N,), dtype=torch.int64, device=device)  # first crossing step (>=1)

    # Pre-generate pulse sequences
    if rng is None:
        rng = np.random.default_rng()

    n_pulses_max = (n_max + steps_per_pulse - 1) // steps_per_pulse
    s_np = np.empty((N, n_pulses_max), dtype=np.float32)
    for i in range(N):
        s_np[i, :] = generate_pulse_sides(rng, n_pulses_max, p_success=p_success)
    s = torch.from_numpy(s_np).to(device=device, dtype=dtype)

    # Time loop
    for t in range(n_max):
        active = (~hit) & (t < n_steps)
        if not torch.any(active):
            break

        noise = torch.randn((N,), device=device, dtype=dtype) * (sigma * sqrt_dt)
        a = a + (-lam * a) * dt + noise

        # Pulse kick
        if (t % steps_per_pulse) == 0:
            p_idx = t // steps_per_pulse
            a = a + v * s[:, p_idx] * active.to(dtype)

        # Bound crossing
        hit_upper = active & (a >= B)
        hit_lower = active & (a <= 0.0)
        newly_hit = hit_upper | hit_lower

        if torch.any(newly_hit):
            # record first hit time (step index is t+1 because we updated a at this step)
            hit_step = torch.where(newly_hit, torch.full_like(hit_step, t + 1), hit_step)
            choice = torch.where(hit_upper, torch.ones_like(choice), choice)
            choice = torch.where(hit_lower, torch.zeros_like(choice), choice)
            hit = hit | newly_hit

    outcome = choice.clone()  # for hits this is already 0/1

    not_hit = ~hit
    if torch.any(not_hit):
        # censoring time in steps (decision window length)
        end_step = torch.clamp(n_steps, min=0)
        hit_step = torch.where(not_hit, end_step, hit_step)

        # mark invalid trials as category 2
        outcome = torch.where(not_hit, torch.full_like(outcome, 2), outcome)

    # RT always defined (hit time or censoring time) + non-decision time
    rt = (t_nd + hit_step.to(dtype) * dt).clamp(1e-6, float(T_MAX))

    x = torch.stack([rt.to(dtype), outcome.to(dtype)], dim=-1)  # (N,2)
    return x


def rt_choice_model_simulator(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_sensory: float = 1.0,
    p_success: float = DEFAULT_P_SUCCESS,
) -> tuple[float, int]:
    """
    Single-trial NumPy API. Returns (rt, choice) with choice in {0,1}.
    """
    th = torch.tensor(theta, dtype=torch.float32).view(1, 5)
    x = _simulate_rt_choice_batch_torch(
        th,
        mu_sensory=float(mu_sensory),
        p_success=float(p_success),
        rng=rng,
    )
    rt = float(x[0, 0].item())
    choice = int(x[0, 1].item())
    return rt, choice


def rt_choice_model_simulator_torch(
    theta: Tensor,
    rng: np.random.Generator | None = None,
    *,
    mu_sensory: float = 1.0,
    p_success: float = DEFAULT_P_SUCCESS,
) -> Tensor:
    """
    SBI-friendly simulator.

    Input:
      theta: (batch,5) or (5,) torch tensor

    Output:
      x: (batch,2) float32 tensor with columns [rt, choice] where choice in {0.,1.}.
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 5:
        raise ValueError(f"Expected theta shape (N,5) or (5,), got {tuple(theta.shape)}")

    return _simulate_rt_choice_batch_torch(
        theta,
        mu_sensory=float(mu_sensory),
        p_success=float(p_success),
        rng=rng,
    ).to(torch.float32)


def simulate_session_data_rt_choice(
    theta_true: Tensor,
    num_trials: int,
    rng: np.random.Generator | None = None,
    *,
    mu_sensory: float = 1.0,
    p_success: float = DEFAULT_P_SUCCESS,
) -> Tensor:
    """
    Simulate IID trials for one 'session': returns (num_trials,2) [rt,choice].
    """
    theta_true = theta_true.view(1, -1).to(torch.float32)
    theta_rep = theta_true.repeat(num_trials, 1)
    return rt_choice_model_simulator_torch(
        theta_rep,
        rng=rng,
        mu_sensory=mu_sensory,
        p_success=p_success,
    )
