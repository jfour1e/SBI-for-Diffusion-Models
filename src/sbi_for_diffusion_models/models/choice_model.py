from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..constants import T_MAX, PULSE_INTERVAL, DT_CHOICE
from sbi_for_diffusion_models.run_config import RUN_CONFIG_PARAMS, RunConfig
cfg = RUN_CONFIG_PARAMS

@dataclass(frozen=True)
class ChoiceModelParams:
    a0_frac: float
    lam: float
    v: float
    B: float
    t_nd: float

    @staticmethod
    def from_theta(theta: np.ndarray) -> "ChoiceModelParams":
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
        t_nd = float(np.clip(t_nd, 0.0, T_MAX - 1e-6))

        return ChoiceModelParams(a0_frac=a0, lam=lam, v=v, B=B, t_nd=t_nd)

def generate_pulse_sides(
    rng: np.random.Generator,
    n_pulses: int,
    *,
    p_success: float = cfg.P_SUCCESS,
) -> np.ndarray:
    """
    Returns s_seq shape (n_pulses,), values +/-1.
    Correct side is random 50/50; each pulse matches correct w.p. p_success.
    """
    if n_pulses <= 0:
        return np.zeros((0,), dtype=np.float32)

    p_success = float(np.clip(p_success, 0.0, 1.0))
    correct_side = 1.0 if rng.random() < 0.5 else -1.0
    is_correct = rng.random(size=n_pulses) < p_success
    s_seq = np.where(is_correct, correct_side, -correct_side).astype(np.float32)
    return s_seq

def _simulate_choice_batch_torch(
    theta: Tensor,
    *,
    mu_sensory: float,
    p_success: float,
    rng: Optional[np.random.Generator],
    resample_invalid: bool,
    max_resamples: int,
) -> Tensor:
    """
    theta: (N,5) float32/float64 torch tensor on CPU or GPU
    returns: (N,) int64 choices in {-1,0,1}
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
    n_max = int(np.floor(T_MAX / dt))
    steps_per_pulse = max(int(np.round(PULSE_INTERVAL / dt)), 1)

    # Decision window per trial in steps
    n_steps = torch.floor((float(T_MAX) - t_nd) / dt).to(torch.int64).clamp(0, n_max)

    # Start point
    a0 = a0_frac * B

    sigma = float(mu_sensory)

    # Helper to run one stochastic pass (used also for resampling invalid trials)
    def one_pass(mask_trials: Optional[Tensor] = None) -> Tensor:
        """
        Returns choices for either all trials or a subset (mask_trials indices).
        """
        if mask_trials is None:
            idx = None
            a = a0.clone()
            lam_ = lam
            v_ = v
            B_ = B
            n_steps_ = n_steps
            N_ = N
        else:
            idx = mask_trials
            a = a0[idx].clone()
            lam_ = lam[idx]
            v_ = v[idx]
            B_ = B[idx]
            n_steps_ = n_steps[idx]
            N_ = a.shape[0]

        # Outcomes initialized to -1 (invalid)
        choice = torch.full((N_,), -1, dtype=torch.int64, device=device)
        hit = torch.zeros((N_,), dtype=torch.bool, device=device)

        # Pre-generate pulse side sequence per trial/pulse as CPU numpy then move to device.
        # This is simple and robust; for speed lets replace with torch RNGG later
        if rng is None:
            local_rng = np.random.default_rng()
        else:
            local_rng = rng

        n_pulses_max = (n_max + steps_per_pulse - 1) // steps_per_pulse

        # s_seq: (N_, n_pulses_max) in +/-1
        s_np = np.empty((N_, n_pulses_max), dtype=np.float32)
        for i in range(N_):
            s_np[i, :] = generate_pulse_sides(local_rng, n_pulses_max, p_success=p_success)
        s = torch.from_numpy(s_np).to(device=device, dtype=dtype)

        # Time loop: vectorized across trials
        # Only update trials that are still active (t < n_steps and not hit).
        sqrt_dt = np.sqrt(dt)

        for t in range(n_max):
            active = (~hit) & (t < n_steps_)
            if not torch.any(active):
                break

            # diffusion
            noise = torch.randn((N_,), device=device, dtype=dtype) * (sigma * sqrt_dt)

            # leak + noise
            a = a + (-lam_ * a) * dt + noise

            # pulse kick if on a pulse step
            if (t % steps_per_pulse) == 0:
                p_idx = t // steps_per_pulse  # 0-based pulse index
                # Only apply pulses for trials whose decision window includes this time
                pulse_active = active & (t < n_steps_)
                if torch.any(pulse_active):
                    a = a + v_ * s[:, p_idx] * pulse_active.to(dtype)

            # check bounds for newly hitting trials
            hit_upper = active & (a >= B_)
            hit_lower = active & (a <= 0.0)

            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                choice = torch.where(hit_upper, torch.ones_like(choice), choice)
                choice = torch.where(hit_lower, torch.zeros_like(choice), choice)
                hit = hit | newly_hit

        return choice

    # First pass
    out = one_pass()

    if resample_invalid:
        invalid = out < 0
        n_try = 0
        while bool(invalid.any()) and n_try < max_resamples:
            idx = torch.where(invalid)[0]
            out2 = one_pass(idx)
            out[idx] = out2
            invalid = out < 0
            n_try += 1

    return out


# public api
def choice_model_simulator(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_sensory: float = 1.0,
    p_success: float = cfg.P_SUCCESS,
) -> int:
    """
    Single-trial NumPy API. Returns {-1,0,1}.
    """
    th = torch.tensor(theta, dtype=torch.float32).view(1, 5)
    out = _simulate_choice_batch_torch(
        th,
        mu_sensory=float(mu_sensory),
        p_success=float(p_success),
        rng=rng,
        resample_invalid=False,
        max_resamples=0,
    )
    return int(out.item())


def choice_model_simulator_torch(
    theta: Tensor,
    rng: np.random.Generator | None = None,
    *,
    mu_sensory: float = 1.0,
    p_success: float = cfg.P_SUCCESS,
    resample_invalid: bool = False,
    max_resamples: int = 50,
) -> Tensor:
    """
    SBI-friendly simulator.

    Input:
      theta: (batch,5) or (5,) torch tensor

    Output:
      x: (batch,1) float32 tensor with values {0.,1.} and optionally -1. for invalid.

    If resample_invalid=True:
      resample noise/stimulus up to max_resamples times for invalid trials.
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 5:
        raise ValueError(f"Expected theta shape (N,5) or (5,), got {tuple(theta.shape)}")

    choices = _simulate_choice_batch_torch(
        theta,
        mu_sensory=float(mu_sensory),
        p_success=float(p_success),
        rng=rng,
        resample_invalid=bool(resample_invalid),
        max_resamples=int(max_resamples),
    )

    # Return as float32 (N,1)
    return choices.to(dtype=torch.float32).view(-1, 1)
