from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..constants import T_MAX, PULSE_INTERVAL, DT_CHOICE
from .choice_model import generate_pulse_sides
from ..run_config import RUN_CONFIG_PARAMS
cfg = RUN_CONFIG_PARAMS

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
    
    
def pulse_schedule(*, dt: float = float(DT_CHOICE)) -> Tuple[int, int]:
    """
    Returns (n_max, steps_per_pulse) for the RT-choice simulator time grid.

    - n_max: total number of Euler steps in [0, T_MAX]
    - steps_per_pulse: number of Euler steps between successive pulses (>=1)
    """
    n_max = int(np.floor(float(T_MAX) / float(dt)))
    steps_per_pulse = max(int(np.round(float(PULSE_INTERVAL) / float(dt))), 1)
    return n_max, steps_per_pulse


def n_pulses_max_from_schedule(n_max: int, steps_per_pulse: int) -> int:
    """Maximum number of pulse slots for a trial of length n_max steps."""
    return (int(n_max) + int(steps_per_pulse) - 1) // int(steps_per_pulse)


def max_num_pulses() -> int:
    """Maximum number of pulses in a trial of duration T_MAX."""
    return int(float(T_MAX) / float(PULSE_INTERVAL))


def generate_pulse_matrix_numpy(
    rng: np.random.Generator,
    n_trials: int,
    n_pulses: int,
    *,
    p_success: float = cfg.P_SUCCESS,
) -> np.ndarray:
    """
    Generate a realized pulse-side matrix s with shape (n_trials, n_pulses), values in {+1,-1}.

    This is intentionally *outside* the simulator so you can:
      - save stimulus per trial,
      - condition on stimulus in inference,
      - reuse the exact same s across repeated likelihood calls.

    Notes
    -----
    This uses the same logic as `choice_model.generate_pulse_sides`:
      - correct side is chosen 50/50 per trial,
      - each pulse matches the correct side with probability p_success.
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


def as_pulse_tensor(
    pulse_sides: Union[np.ndarray, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert pulse_sides to a (N, P) torch.Tensor on the desired device."""
    if isinstance(pulse_sides, Tensor):
        s = pulse_sides
    else:
        s = torch.from_numpy(np.asarray(pulse_sides))
    if s.ndim == 1:
        s = s.view(1, -1)
    if s.ndim != 2:
        raise ValueError(f"pulse_sides must have shape (N,P) or (P,), got {tuple(s.shape)}")
    return s.to(device=device, dtype=dtype)


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
    noise_std = sigma * torch.sqrt(var_factor.clamp_min(1e-30))
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
    device: torch.device,
    dtype: torch.dtype,
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
    N = a0_frac.shape[0]

    a = a0_frac * B
    hit = torch.zeros(N, dtype=torch.bool, device=device)
    choice = torch.zeros(N, dtype=torch.int64, device=device)
    hit_interval = torch.zeros(N, dtype=torch.int64, device=device)
    a_before_hit = torch.zeros(N, dtype=dtype, device=device)
    hit_from_pulse = torch.zeros(N, dtype=torch.bool, device=device)

    all_noise = torch.randn((P_max, N), device=device, dtype=dtype)

    for k in range(P_max):
        active = (~hit) & (k < n_intervals)
        if not torch.any(active):
            break

        a_prev = a.clone()

        # --- exact OU step for one full pulse interval ---
        a = torch.where(active, a * decay + noise_std * all_noise[k], a)

        # boundary check after OU step
        hit_upper = active & (a >= B)
        hit_lower = active & (a <= 0.0)
        newly_hit = hit_upper | hit_lower
        if torch.any(newly_hit):
            hit_interval = torch.where(newly_hit, torch.full_like(hit_interval, k + 1), hit_interval)
            choice = torch.where(hit_upper, torch.ones_like(choice), choice)
            choice = torch.where(hit_lower, torch.zeros_like(choice), choice)
            a_before_hit = torch.where(newly_hit, a_prev, a_before_hit)
            hit_from_pulse = torch.where(newly_hit, torch.zeros_like(hit_from_pulse), hit_from_pulse)
            hit = hit | newly_hit

        # --- pulse kick (pulse k arrives at time (k+1)*delta) ---
        still_active = (~hit) & (k < n_intervals)
        if torch.any(still_active):
            a = torch.where(still_active, a + v * s[:, k], a)

            hit_upper = still_active & (a >= B)
            hit_lower = still_active & (a <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_interval = torch.where(newly_hit, torch.full_like(hit_interval, k + 1), hit_interval)
                choice = torch.where(hit_upper, torch.ones_like(choice), choice)
                choice = torch.where(hit_lower, torch.zeros_like(choice), choice)
                a_before_hit = torch.where(newly_hit, a_prev, a_before_hit)
                hit_from_pulse = torch.where(newly_hit, torch.ones_like(hit_from_pulse), hit_from_pulse)
                hit = hit | newly_hit

    return hit, choice, hit_interval, a_before_hit, hit_from_pulse


# Refinement sub-steps used only inside the single crossing interval.
# resolution = PULSE_INTERVAL / _DEFAULT_N_REFINE  (default 0.1 ms)
_DEFAULT_N_REFINE = 1000


def _simulate_rt_choice_batch_torch(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: float = cfg.P_SUCCESS,
    rng: Optional[np.random.Generator] = None,
    max_resamples: int = 100,
    n_refine: int = _DEFAULT_N_REFINE,
) -> Tensor:
    """
    Batch simulator using adaptive-resolution exact OU transitions.

    1. **Coarse pass** (~80 iterations): one exact OU step per pulse
       interval to identify *which* interval each trial crosses in.
    2. **Refinement** (n_refine iterations, only for the crossing
       interval): re-simulate that single interval at high resolution
       to pinpoint the crossing time.

    Timeout trials are resampled with fresh noise until every trial
    produces a decision.

    Parameters
    ----------
    n_refine : int
        Sub-steps used to refine the crossing interval.
        RT resolution = PULSE_INTERVAL / n_refine  (default 0.1 ms).

    theta : (N, 5)  [a0_frac, lam, v, B, t_nd]
    returns : (N, 2)  [rt, choice]  with choice in {0, 1}
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

    delta = float(PULSE_INTERVAL)
    sigma = float(mu_sensory)
    P_max = max_num_pulses()

    # decision window in full pulse intervals
    n_intervals = torch.floor((float(T_MAX) - t_nd) / delta).to(torch.int64).clamp(0, P_max)

    # coarse OU params (one full pulse interval)
    decay_coarse, noise_std_coarse = _ou_transition_params(lam, delta, sigma)

    # pulse sides
    if pulse_sides is None:
        if rng is None:
            rng = np.random.default_rng()
        s_np = generate_pulse_matrix_numpy(rng, N, P_max, p_success=p_success)
        s = torch.from_numpy(s_np).to(device=device, dtype=dtype)
    else:
        s = as_pulse_tensor(pulse_sides, device=device, dtype=dtype)
        if s.shape[0] == 1 and N > 1:
            s = s.expand(N, -1)
        if s.shape[0] != N:
            raise ValueError(
                f"pulse_sides first dim must match batch size N={N} "
                f"(or be 1 for broadcast), got {s.shape[0]}"
            )
        if s.shape[1] < P_max:
            raise ValueError(
                f"pulse_sides has P={s.shape[1]} pulses but simulator "
                f"needs at least {P_max} for T_MAX={T_MAX}s"
            )
        s = s[:, :P_max]

    # ---- coarse pass ----
    hit, choice, hit_interval, a_before_hit, hit_from_pulse = _run_coarse_ou_loop(
        a0_frac, v, B, decay_coarse, noise_std_coarse, s,
        n_intervals, P_max, device, dtype,
    )

    # ---- resample timeout trials (same theta + pulses, fresh noise) ----
    not_hit = ~hit
    for _ in range(max_resamples):
        if not torch.any(not_hit):
            break
        idx = torch.where(not_hit)[0]
        h, c, hi, abh, hfp = _run_coarse_ou_loop(
            a0_frac[idx], v[idx], B[idx],
            decay_coarse[idx], noise_std_coarse[idx], s[idx],
            n_intervals[idx], P_max, device, dtype,
        )
        hit[idx] = h
        choice[idx] = c
        hit_interval[idx] = hi
        a_before_hit[idx] = abh
        hit_from_pulse[idx] = hfp
        not_hit = ~hit

    # fallback for any remaining timeouts
    if torch.any(not_hit):
        n_left = not_hit.sum().item()
        print(f"Warning: {n_left} trials still timed out after {max_resamples} resamples.")
        choice[not_hit] = torch.randint(0, 2, (n_left,), device=device)
        hit_interval[not_hit] = n_intervals[not_hit]

    # ---- compute decision time ----
    # default: crossing at the end of the interval (used for pulse-kick
    # crossings and as the fallback for timeouts)
    decision_time = hit_interval.to(dtype) * delta

    # ---- refine OU-step crossings ----
    ou_mask = hit & (~hit_from_pulse)
    if torch.any(ou_mask) and n_refine > 1:
        idx = torch.where(ou_mask)[0]
        M = idx.shape[0]
        k_cross = hit_interval[idx] - 1          # 0-based interval index

        # per-trial pulse kick for this interval
        pulse_val = v[idx] * s[idx, k_cross]

        # fine OU params
        sub_delta = delta / n_refine
        decay_fine, noise_std_fine = _ou_transition_params(lam[idx], sub_delta, sigma)

        a_r = a_before_hit[idx].clone()
        hit_r = torch.zeros(M, dtype=torch.bool, device=device)
        choice_r = torch.zeros(M, dtype=torch.int64, device=device)
        hit_substep = torch.full((M,), n_refine, dtype=torch.int64, device=device)

        ref_noise = torch.randn((n_refine, M), device=device, dtype=dtype)

        for t in range(n_refine):
            active_r = ~hit_r
            if not torch.any(active_r):
                break

            a_r = torch.where(active_r, a_r * decay_fine + noise_std_fine * ref_noise[t], a_r)

            hit_upper = active_r & (a_r >= B[idx])
            hit_lower = active_r & (a_r <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_substep = torch.where(newly_hit, torch.full_like(hit_substep, t + 1), hit_substep)
                choice_r = torch.where(hit_upper, torch.ones_like(choice_r), choice_r)
                choice_r = torch.where(hit_lower, torch.zeros_like(choice_r), choice_r)
                hit_r = hit_r | newly_hit

        # apply pulse for trials still active after all OU sub-steps
        still_active = ~hit_r
        if torch.any(still_active):
            a_r = torch.where(still_active, a_r + pulse_val, a_r)
            hit_upper = still_active & (a_r >= B[idx])
            hit_lower = still_active & (a_r <= 0.0)
            newly_hit = hit_upper | hit_lower
            if torch.any(newly_hit):
                hit_substep = torch.where(newly_hit, torch.full_like(hit_substep, n_refine), hit_substep)
                choice_r = torch.where(hit_upper, torch.ones_like(choice_r), choice_r)
                choice_r = torch.where(hit_lower, torch.zeros_like(choice_r), choice_r)
                hit_r = hit_r | newly_hit

        # write back refined decision times and choices
        decision_time[idx] = k_cross.to(dtype) * delta + hit_substep.to(dtype) * sub_delta
        choice[idx[hit_r]] = choice_r[hit_r]

    # RT = non-decision time + decision time
    rt = (t_nd + decision_time).clamp(1e-6, float(T_MAX))
    x = torch.stack([rt, choice.to(dtype)], dim=-1)
    return x


def rt_choice_model_simulator(
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_sensory: float = 1.0,
    pulse_sides: Optional[Union[np.ndarray, Tensor]] = None,
    p_success: float = cfg.P_SUCCESS,
) -> tuple[float, int]:
    """
    Single-trial NumPy API.

    If `pulse_sides` is provided (shape (P,) or (1,P)), the simulator is *conditioned* on that
    realized stimulus sequence. If it is None, stimulus is sampled internally (marginalized).
    """
    th = torch.tensor(theta, dtype=torch.float32).view(1, 5)
    x = _simulate_rt_choice_batch_torch(
        th,
        mu_sensory=float(mu_sensory),
        pulse_sides=pulse_sides,
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
    pulse_sides: Optional[Union[np.ndarray, Tensor]] = None,
    p_success: float = cfg.P_SUCCESS,
) -> Tensor:
    """
    SBI-friendly simulator.

    Input:
      theta: (batch,5) or (5,) torch tensor

    Output:
      x: (batch,2) float32 tensor with columns [rt, choice] where choice in {0.,1.}.

    Conditioning on stimulus:
      Provide `pulse_sides` with shape (batch,P) (or (P,) / (1,P) to broadcast).
      This prevents "integrating out" the stimulus during simulation.
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 5:
        raise ValueError(f"Expected theta shape (N,5) or (5,), got {tuple(theta.shape)}")

    return _simulate_rt_choice_batch_torch(
        theta,
        mu_sensory=float(mu_sensory),
        pulse_sides=pulse_sides,
        p_success=float(p_success),
        rng=rng,
    ).to(torch.float32)


def simulate_session_data_rt_choice(
    theta_true: Tensor,
    num_trials: int,
    rng: np.random.Generator | None = None,
    *,
    mu_sensory: float = 1.0,
    pulse_sides: Optional[Union[np.ndarray, Tensor]] = None,
    p_success: float = cfg.P_SUCCESS,
    return_pulse_sides: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Simulate IID trials for one 'session': returns (num_trials,2) [rt,choice].

    Recommended conditioning workflow:
      1) Generate stimulus externally via `generate_pulse_matrix_numpy`.
      2) Pass it in as `pulse_sides=...` to ensure the simulator conditions on the realized stimulus.

    If return_pulse_sides=True, returns (x, s) where s is (num_trials, P) torch.float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_true = theta_true.view(1, -1).to(torch.float32)
    theta_rep = theta_true.repeat(num_trials, 1)

    # If not provided, we generate stimulus *outside* the simulator body (still marginal unless you save it).
    if pulse_sides is None:
        P = max_num_pulses()
        s_np = generate_pulse_matrix_numpy(rng, num_trials, P, p_success=p_success)
        pulse_sides = s_np

    x = rt_choice_model_simulator_torch(
        theta_rep,
        rng=rng,  # only used if pulse_sides is None (should not happen here)
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,
    )

    if return_pulse_sides:
        s_t = as_pulse_tensor(pulse_sides, device=x.device, dtype=torch.float32)
        return x, s_t
    return x

# helper functions
def pack_x_rt_choice(rt_choice: torch.Tensor, *, log_rt: bool) -> torch.Tensor:
    """
    MNLE expects x to contain continuous component(s) and then a discrete/categorical
    component in the last dimension. Choice values are in {0,1} stored as float.
    """
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)
