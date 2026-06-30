from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..run_config import RUN_CONFIG_PARAMS, T_MAX, PULSE_INTERVAL
cfg = RUN_CONFIG_PARAMS


def max_num_pulses() -> int:
    return int(float(T_MAX) / float(PULSE_INTERVAL))


def as_pulse_tensor(
    pulse_sides: Union[np.ndarray, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
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
    p_success: Union[float, Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    return_correct_side: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Torch-native pulse generator in {-1, +1} with shape (n_trials, n_pulses).

    p_success may be a scalar or a 1-D tensor of shape (n_trials,), supporting
    mixed experimental conditions within a single batch.

    If `return_correct_side=True`, also returns the (n_trials,) intended-correct
    side in {-1, +1} (needed for autoregressive training to define 'previous
    trial was correct').
    """
    if n_trials < 0 or n_pulses < 0:
        raise ValueError("n_trials and n_pulses must be >= 0")
    if n_trials == 0 or n_pulses == 0:
        s = torch.empty((n_trials, n_pulses), device=device, dtype=dtype)
        if return_correct_side:
            return s, torch.empty((n_trials,), device=device, dtype=dtype)
        return s

    correct_side = torch.randint(
        low=0, high=2, size=(n_trials,), device=device, generator=generator
    )
    correct_side = (1 - 2 * correct_side).to(dtype)

    if isinstance(p_success, Tensor):
        if p_success.shape != (n_trials,):
            raise ValueError(
                f"p_success tensor must have shape ({n_trials},); got {tuple(p_success.shape)}"
            )
        p_thresh = p_success.to(device=device, dtype=dtype).unsqueeze(1)
    else:
        p_thresh = float(p_success)

    is_correct = torch.rand((n_trials, n_pulses), device=device, generator=generator) < p_thresh
    s = torch.where(is_correct, correct_side[:, None], -correct_side[:, None]).to(dtype)
    if return_correct_side:
        return s, correct_side
    return s


def generate_pulses_human_torch(
    n_trials: int,
    n_pulses: int,
    *,
    p_success: Union[float, Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    return_correct_side: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Human pulse generator: two independent Bernoullis per bin.

    Per bin: P(right_flash=1) = p_success, P(left_flash=1) = 1 - p_success,
    sampled independently. Returns s_k = right - left in {-1, 0, +1}.

    This differs from `generate_pulses_torch` (marmoset): the marmoset random-
    probability scheme uses a single coin flip per bin (XOR — exactly one side
    flashes), while the human task generates the two sides independently, so
    each bin can have neither, one, or both sides flash.

    correct_side is derived from the assigned `p_success` (the intended bias),
    NOT from realized counts: +1 if p > 0.5, -1 if p < 0.5, random at p = 0.5.
    """
    if n_trials < 0 or n_pulses < 0:
        raise ValueError("n_trials and n_pulses must be >= 0")
    if n_trials == 0 or n_pulses == 0:
        s = torch.empty((n_trials, n_pulses), device=device, dtype=dtype)
        if return_correct_side:
            return s, torch.empty((n_trials,), device=device, dtype=dtype)
        return s

    if isinstance(p_success, Tensor):
        if p_success.shape != (n_trials,):
            raise ValueError(
                f"p_success tensor must have shape ({n_trials},); got {tuple(p_success.shape)}"
            )
        p_R = p_success.to(device=device, dtype=dtype)
    else:
        p_R = torch.full((n_trials,), float(p_success), device=device, dtype=dtype)
    p_L = 1.0 - p_R  # complementary by Amanda's description; total rate = 1 event/bin

    # Independent per-side Bernoullis per bin
    u_R = torch.rand((n_trials, n_pulses), device=device, generator=generator)
    u_L = torch.rand((n_trials, n_pulses), device=device, generator=generator)
    right_flash = (u_R < p_R.unsqueeze(1)).to(dtype)
    left_flash = (u_L < p_L.unsqueeze(1)).to(dtype)
    s = right_flash - left_flash  # in {-1, 0, +1}

    if return_correct_side:
        # +1 if assigned p > 0.5, -1 if < 0.5, random tie-break at exactly 0.5
        tie = (p_R == 0.5)
        sign = torch.where(p_R > 0.5, torch.ones_like(p_R), -torch.ones_like(p_R))
        if tie.any():
            random_sign = (
                torch.randint(0, 2, (n_trials,), device=device, generator=generator).to(dtype)
                * 2.0
                - 1.0
            )
            sign = torch.where(tie, random_sign, sign)
        return s, sign
    return s


def sample_p_success_human_cascade(
    n_trials: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    p_random_continuous_uniform: Tuple[float, float] = (0.0, 1.0),
) -> Tensor:
    """Sample per-trial p_R (probflashright) using the human task's coin-flip cascade.

    Cascade (each branch with 50/50 coin):
      Flip 1: assign 70/30  OR  go to Flip 2
      Flip 2: assign 70/30  OR  go to Flip 3
      Flip 3: assign 60/40  OR  go to Flip 4
      Flip 4: assign 60/40  OR  draw p_R ~ Uniform(0, 1)

    Resulting (approximate) marginal:
      P(p_R ∈ {0.3, 0.7}) = 0.75   (70/30 stages)
      P(p_R ∈ {0.4, 0.6}) = 0.1875 (60/40 stages)
      P(p_R continuous)   = 0.0625

    Within each discrete stage, p_R is randomly assigned to the "harder" or "easier"
    side (50/50), so 70/30 becomes p_R=0.7 half the time and p_R=0.3 the other half.
    """
    if n_trials <= 0:
        return torch.empty((0,), device=device, dtype=dtype)

    # Sample stage assignment via the cascade
    f1 = torch.rand((n_trials,), device=device, generator=generator)
    f2 = torch.rand((n_trials,), device=device, generator=generator)
    f3 = torch.rand((n_trials,), device=device, generator=generator)
    f4 = torch.rand((n_trials,), device=device, generator=generator)

    # category: 0 = 70/30 (from flip 1 or 2), 1 = 60/40 (from flip 3 or 4), 2 = continuous
    # at-flip-1 70/30 (prob 0.5)
    # at-flip-2 70/30 (prob 0.5 * 0.5 = 0.25)
    # at-flip-3 60/40 (prob 0.5^3 = 0.125)
    # at-flip-4 60/40 (prob 0.5^4 = 0.0625)
    # at-flip-4 continuous (prob 0.5^4 = 0.0625)
    cat = torch.full((n_trials,), 2, device=device, dtype=torch.int64)  # default continuous
    # First passing flip wins:
    at1_70 = f1 < 0.5
    at2_70 = (~at1_70) & (f2 < 0.5)
    at3_60 = (~at1_70) & (~(f2 < 0.5)) & (f3 < 0.5)
    at4_60 = (~at1_70) & (~(f2 < 0.5)) & (~(f3 < 0.5)) & (f4 < 0.5)
    # the rest (~6.25%) are continuous
    cat[at1_70 | at2_70] = 0
    cat[at3_60 | at4_60] = 1

    # For discrete stages, random side assignment (each side preferred 50/50)
    side_coin = torch.rand((n_trials,), device=device, generator=generator)

    p_R = torch.empty((n_trials,), device=device, dtype=dtype)
    # 70/30
    mask = cat == 0
    p_R[mask] = torch.where(side_coin[mask] < 0.5, torch.tensor(0.7, device=device, dtype=dtype),
                            torch.tensor(0.3, device=device, dtype=dtype))
    # 60/40
    mask = cat == 1
    p_R[mask] = torch.where(side_coin[mask] < 0.5, torch.tensor(0.6, device=device, dtype=dtype),
                            torch.tensor(0.4, device=device, dtype=dtype))
    # Continuous
    mask = cat == 2
    lo, hi = float(p_random_continuous_uniform[0]), float(p_random_continuous_uniform[1])
    p_R[mask] = lo + (hi - lo) * torch.rand(int(mask.sum().item()), device=device, generator=generator, dtype=dtype)

    return p_R


def mask_unperceived_pulses(
    pulse_sides: Tensor,
    rt: Tensor,
    pulse_interval: float = float(PULSE_INTERVAL),
) -> Tensor:
    N, P = pulse_sides.shape
    # Pulse k arrives at t = (k+1)*pulse_interval
    pulse_times = (torch.arange(P, device=pulse_sides.device, dtype=torch.float32) + 1) * pulse_interval
    unperceived = pulse_times.unsqueeze(0) > rt.to(pulse_sides.device).unsqueeze(1)
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
    """Exact OU decay and noise std for a step of size dt."""
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

        a_start = a

        a = torch.where(active, a * decay + noise_std * all_noise[k], a)

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
    s: Tensor,
    *,
    mu_sensory: float,
    dt_internal: float,
    pulse_interval: float,
    T_MAX: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    device = a0_frac.device
    dtype = a0_frac.dtype
    N = a0_frac.shape[0]

    delta = float(pulse_interval)
    dt0 = float(dt_internal)

    # Make dt divide the pulse interval exactly to avoid drift.
    steps_per_pulse = max(1, int(round(delta / dt0)))
    dt = delta / steps_per_pulse

    P_max = s.shape[1]

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


def simulate_rt_choice_batch(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: Union[float, Tensor] = cfg.P_SUCCESS,
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
    lam = theta[:, 1].abs()
    v = theta[:, 2].abs()
    B = theta[:, 3].abs().clamp_min(1e-6)
    tau = theta[:, 4].clamp(0.0, float(T_MAX) - 1e-6)

    delta = float(PULSE_INTERVAL)
    P_max = max_num_pulses()

    if pulse_sides is None:
        s = generate_pulses_torch(
            n_trials=N,
            n_pulses=P_max,
            p_success=p_success,
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


def _insert_zero_lam(theta: Tensor) -> Tensor:
    """Insert a zero lam column at index 1 of theta (no-leak variants)."""
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    zeros = torch.zeros((theta.shape[0], 1), device=theta.device, dtype=theta.dtype)
    return torch.cat([theta[:, :1], zeros, theta[:, 1:]], dim=1)


def simulate_rt_choice_batch_noleak(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: Union[float, Tensor] = cfg.P_SUCCESS,
    pulse_generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """No-leak variant: theta = [a0, v, B, tau]. Equivalent to lam=0 (pure Brownian)."""
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 4:
        raise ValueError(f"Expected theta shape (N,4) or (4,), got {tuple(theta.shape)}")
    return simulate_rt_choice_batch(
        _insert_zero_lam(theta),
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,
        pulse_generator=pulse_generator,
    )


def apply_ar_bias_to_a0(
    a0: Tensor,
    w_corr: Tensor,
    w_err: Tensor,
    prev_choice_signed: Tensor,
    prev_outcome_signed: Tensor,
) -> Tensor:
    """Win-stay/lose-shift bias on the starting point.

    delta = (w_corr if prev outcome correct else w_err) * prev_choice_signed
    a0_eff = clamp(a0 + delta, 0, 1)
    First trials (prev_choice_signed == 0) get zero shift naturally.
    """
    is_correct = prev_outcome_signed > 0
    w_signed = torch.where(is_correct, w_corr, w_err)
    delta = w_signed * prev_choice_signed
    return (a0 + delta).clamp(0.0, 1.0)


def simulate_rt_choice_batch_ar(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: Union[float, Tensor] = cfg.P_SUCCESS,
    pulse_generator: Optional[torch.Generator] = None,
    prev_choice_signed: Optional[Tensor] = None,
    prev_outcome_signed: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """AR variant: theta = [a0, lam, v, B, tau, w_corr, w_err].

    prev_choice_signed, prev_outcome_signed: (N,) tensors in {-1, 0, +1}; 0
    means 'no previous trial'. Default to all zeros (no AR effect; matches base).
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 7:
        raise ValueError(f"Expected theta shape (N,7) or (7,), got {tuple(theta.shape)}")

    device = theta.device
    dtype = torch.float32
    theta = theta.to(dtype=dtype)
    N = theta.shape[0]

    if prev_choice_signed is None:
        prev_choice_signed = torch.zeros((N,), device=device, dtype=dtype)
    if prev_outcome_signed is None:
        prev_outcome_signed = torch.zeros((N,), device=device, dtype=dtype)

    a0_eff = apply_ar_bias_to_a0(
        theta[:, 0].clamp(0.0, 1.0),
        theta[:, 5],
        theta[:, 6],
        prev_choice_signed.to(device=device, dtype=dtype),
        prev_outcome_signed.to(device=device, dtype=dtype),
    )

    theta_inner = torch.stack(
        [a0_eff, theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]], dim=1
    )
    return simulate_rt_choice_batch(
        theta_inner,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,
        pulse_generator=pulse_generator,
    )


def simulate_rt_choice_batch_noleak_ar(
    theta: Tensor,
    *,
    mu_sensory: float,
    pulse_sides: Optional[Union[Tensor, np.ndarray]] = None,
    p_success: Union[float, Tensor] = cfg.P_SUCCESS,
    pulse_generator: Optional[torch.Generator] = None,
    prev_choice_signed: Optional[Tensor] = None,
    prev_outcome_signed: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """No-leak AR variant: theta = [a0, v, B, tau, w_corr, w_err]."""
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    if theta.shape[-1] != 6:
        raise ValueError(f"Expected theta shape (N,6) or (6,), got {tuple(theta.shape)}")
    return simulate_rt_choice_batch_ar(
        _insert_zero_lam(theta),
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,
        pulse_generator=pulse_generator,
        prev_choice_signed=prev_choice_signed,
        prev_outcome_signed=prev_outcome_signed,
    )
