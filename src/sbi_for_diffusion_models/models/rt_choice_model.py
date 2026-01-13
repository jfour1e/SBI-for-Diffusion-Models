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

    s = np.empty((n_trials, n_pulses), dtype=np.float32)
    for i in range(n_trials):
        s[i, :] = generate_pulse_sides(rng, n_pulses, p_success=p_success)
    return s


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


def _simulate_rt_choice_batch_torch(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: float = cfg.P_SUCCESS,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    """
    theta: (N,5) torch tensor on CPU or GPU
    returns: (N,2) float32 tensor: [rt, choice] where choice in {0,1,2}
            (2 denotes censoring / no-bound-hit within the decision window)
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
    n_max, steps_per_pulse = pulse_schedule(dt=dt)

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

    # Pulse sides: either provided (conditioning) or generated here (marginalizing stimulus).
    n_pulses_max = n_pulses_max_from_schedule(n_max, steps_per_pulse)

    if pulse_sides is None:
        # NOTE: This path *integrates out* stimulus by sampling it internally.
        # For conditioning on a realized stimulus, generate `pulse_sides` externally and pass it in.
        if rng is None:
            rng = np.random.default_rng()
        s_np = generate_pulse_matrix_numpy(rng, N, n_pulses_max, p_success=p_success)
        s = torch.from_numpy(s_np).to(device=device, dtype=dtype)
    else:
        s = as_pulse_tensor(pulse_sides, device=device, dtype=dtype)
        if s.shape[0] == 1 and N > 1:
            # allow broadcasting a single stimulus across a batch
            s = s.expand(N, -1)
        if s.shape[0] != N:
            raise ValueError(
                f"pulse_sides first dim must match batch size N={N} (or be 1 for broadcast), got {s.shape[0]}"
            )
        if s.shape[1] < n_pulses_max:
            raise ValueError(
                f"pulse_sides has P={s.shape[1]} pulses but simulator needs at least {n_pulses_max} for T_MAX={T_MAX}s"
            )
        # If provided longer than needed, ignore the tail for safety.
        s = s[:, :n_pulses_max]

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
      x: (batch,2) float32 tensor with columns [rt, choice] where choice in {0.,1.,2.}.

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
        n_max, steps_per_pulse = pulse_schedule(dt=float(DT_CHOICE))
        P = n_pulses_max_from_schedule(n_max, steps_per_pulse)
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
    component in the last dimension. We keep choice values in {0,1,2} and store as float,
    but we do *not* apply log to choice.
    """
    rt = rt_choice[:, 0:1].to(torch.float32).clamp_min(1e-6)
    if log_rt:
        rt = torch.log(rt)
    choice = rt_choice[:, 1:2].to(torch.int64)
    return torch.cat([rt, choice.to(torch.float32)], dim=1)
