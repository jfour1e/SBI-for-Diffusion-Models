import torch
from typing import Optional, Callable
import math
import warnings

from sbi_for_diffusion_models.models.rt_choice_model import (
    pack_x_rt_choice,
    generate_pulses_torch,
    mask_unperceived_pulses,
)
from .run_config import T_MAX, PULSE_INTERVAL, MAX_TIMEOUT_TRIES, TIMEOUT_FRAC_ALLOWED

@torch.no_grad()
def simulate_training_sessions(
    prior_theta,
    num_sessions: int,
    num_trials: int,
    *,
    simulate_batch_fn: Callable,
    device: torch.device,
    mu_sensory: float,
    p_success: float,
    P: int,
    log_rt: bool,
    seed: int = 0,
    theta: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate session-level training data for NPE.

    Each session:
      - choose theta (either sampled from prior_theta, or provided via `theta`)
      - generate pulses (T,P)
      - simulate T trials with simulate_batch_fn
      - pack x as [rt, choice]
      - concatenate per-trial features as [rt, choice, pulse_1..pulse_P] 
      - flatten to (T*(2+P),)
    """
    N = int(num_sessions)
    T = int(num_trials)
    P = int(P)
    trial_dim = 2 + P
    NT = N * T

    torch.manual_seed(int(seed))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # infer theta dim from passed in theta or prior draw
    if theta is not None:
        theta_all = torch.as_tensor(theta, device=device, dtype=torch.float32)
        if theta_all.ndim == 1:
            D = theta_all.shape[0]
            theta_all = theta_all.unsqueeze(0).expand(N, -1).contiguous()
        elif theta_all.ndim == 2:
            D = theta_all.shape[1]
            if theta_all.shape[0] != N:
                raise ValueError(
                    f"theta must have shape ({N},{theta_all.shape[1]}); "
                    f"got {tuple(theta_all.shape)}"
                )
        else:
            raise ValueError(
                f"theta must be 1-D or 2-D; got ndim={theta_all.ndim}"
            )
    else:
        # Sample N thetas from the prior
        theta_all = torch.as_tensor(
            prior_theta.sample((N,)),
            device=device,
            dtype=torch.float32,
        )
        if theta_all.ndim == 1:
            theta_all = theta_all.unsqueeze(-1)
        D = theta_all.shape[1]

    # Expand theta to (N*T, D)
    theta_rep = theta_all.repeat_interleave(T, dim=0)  

    pulses = generate_pulses_torch(
        n_trials=NT,
        n_pulses=P,
        p_success=float(p_success),
        device=device,
        dtype=torch.float32,
        generator=gen,
    ) # (NT, P)

    # Simulate all N*T trials in one batched call
    x_raw, hit, _ = simulate_batch_fn(
        theta_rep,
        mu_sensory=float(mu_sensory),
        pulse_sides=pulses,
        p_success=float(p_success),
        pulse_generator=gen,
    )

    # Retry timed-out trials
    tries_used = torch.zeros(NT, device=device, dtype=torch.int64)

    for _ in range(int(MAX_TIMEOUT_TRIES)):
        retry_mask = (~hit) & (tries_used < int(MAX_TIMEOUT_TRIES))
        idx = retry_mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            break

        M = idx.numel()

        pulses_sub = generate_pulses_torch(
            n_trials=M,
            n_pulses=P,
            p_success=float(p_success),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        # re-simulate only the timed-out subset
        x_new, hit_new, _ = simulate_batch_fn(
            theta_rep[idx],
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses_sub,
            p_success=float(p_success),
            pulse_generator=gen,
        )

        x_raw.index_copy_(0, idx, x_new)
        hit.index_copy_(0, idx, hit_new)
        pulses.index_copy_(0, idx, pulses_sub)
        tries_used.index_add_(
            0, idx, torch.ones(M, device=device, dtype=torch.int64),
        )
    
    # per session timeout warning 
    allowed_timeouts = math.ceil(TIMEOUT_FRAC_ALLOWED * T)
    not_hit = ~hit
    not_hit_per_session = not_hit.view(N, T)              # (N, T)
    timeouts_per_session = not_hit_per_session.sum(dim=1)  # (N,)

    bad_sessions = (timeouts_per_session > allowed_timeouts).nonzero(
        as_tuple=False,
    ).squeeze(1)
    for i in bad_sessions.tolist():
        n_to = int(timeouts_per_session[i].item())
        frac = n_to / max(1, T)
        warnings.warn(
            f"[simulate_training_sessions] High timeout rate in session {i}: "
            f"{n_to}/{T} ({frac:.1%}) after {MAX_TIMEOUT_TRIES} retries "
            f"(allowed {TIMEOUT_FRAC_ALLOWED:.0%}). "
            f"theta={theta_all[i].cpu().tolist()}. "
            f"Proceeding with forced T_MAX trials — consider tightening the prior.",
            RuntimeWarning,
            stacklevel=2,
        )
    
    n_total_timeouts = int(not_hit.sum().item())
    if n_total_timeouts > 0:
        idx = not_hit.nonzero(as_tuple=False).squeeze(1)
        M = idx.numel()

        forced_rt = torch.full(
            (M, 1), float(T_MAX), device=device, dtype=torch.float32,
        )
        forced_choice = torch.randint(
            0, 2, (M, 1), device=device, generator=gen, dtype=torch.int64,
        ).to(torch.float32)
        x_raw.index_copy_(0, idx, torch.cat([forced_rt, forced_choice], dim=1))

    # Mask unperceived pulses to 0 
    rt_raw = x_raw[:, 0]  # (NT,)
    pulses = torch.nan_to_num(
        mask_unperceived_pulses(pulses, rt_raw, float(PULSE_INTERVAL)),
        nan=0.0,
    )

    # pack log(rt), choice pairs 
    x_packed = pack_x_rt_choice(x_raw, log_rt=bool(log_rt))  # (NT, 2)

    # Concatenate trial features: [rt, choice, pulse_0 .. pulse_{P-1}]
    trial_features = torch.cat([x_packed, pulses], dim=1)  # (NT, 2+P)

    x_all = trial_features.view(N, T, trial_dim).reshape(N, T * trial_dim)

    return theta_all, x_all


def flatten_observed_session(
    x_o: torch.Tensor,
    pulses_o: torch.Tensor,
    mask_o: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten an observed session into a single row for NPE inference.

    Args:
      x_o: (T,2) packed [rt, choice]
      pulses_o: (T,P)
      mask_o: (T,1) float in {0,1}

    Returns:
      (1, T*(2+P+1))
    """
    trial_features = torch.cat([x_o, pulses_o, mask_o], dim=-1)  # (T, 2+P+1)
    return trial_features.reshape(1, -1).to(torch.float32)