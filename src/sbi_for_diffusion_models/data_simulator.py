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
    warn_on_timeouts: bool = True,
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

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # infer theta dim from passed in theta or prior draw
    if theta is not None:
        theta_all = torch.as_tensor(theta, device=device, dtype=torch.float32)
        if theta_all.ndim == 1:
            theta_all = theta_all.unsqueeze(0).expand(N, -1).contiguous()
        elif theta_all.ndim == 2:
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

    # Expand theta to (N*T, D)
    theta_batch = theta_all[:, None, :].expand(N, T, theta_all.shape[1]).reshape(NT, -1)

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
        theta_batch,
        mu_sensory=float(mu_sensory),
        pulse_sides=pulses,
        p_success=float(p_success),
        pulse_generator=gen,
    )

    for _ in range(int(MAX_TIMEOUT_TRIES)):
        idx = (~hit).nonzero(as_tuple=True)[0]
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
            theta_batch[idx],
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses_sub,
            p_success=float(p_success),
            pulse_generator=gen,
        )

        x_raw.index_copy_(0, idx, x_new)
        hit.index_copy_(0, idx, hit_new)
        pulses.index_copy_(0, idx, pulses_sub)

    # per session timeout warning 
    not_hit = ~hit

    if warn_on_timeouts:
        allowed_timeouts = math.ceil(TIMEOUT_FRAC_ALLOWED * T)
        timeouts_per_session = not_hit.view(N, T).sum(dim=1)
        n_bad = int((timeouts_per_session > allowed_timeouts).sum().item())
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} sessions exceeded timeout threshold after retries.",
                RuntimeWarning,
                stacklevel=2,
            )

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
    
    idx = not_hit.nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        x_raw[idx, 0] = float(T_MAX)
        x_raw[idx, 1] = torch.randint(
            0, 2, (idx.numel(),), device=device, generator=gen
        ).to(x_raw.dtype)

    # Mask unperceived pulses to 0 
    rt_raw = x_raw[:, 0]  # (NT,)
    pulses = mask_unperceived_pulses(pulses, rt_raw, float(PULSE_INTERVAL))
    pulses = torch.nan_to_num(pulses, nan=0.0)

    # pack log(rt), choice pairs 
    x_packed = pack_x_rt_choice(x_raw, log_rt=bool(log_rt))  # (NT, 2)

    x_all = torch.empty((N, T, trial_dim), device=device, dtype=torch.float32)
    x_all[..., :2] = x_packed.view(N, T, 2)
    x_all[..., 2:] = pulses.view(N, T, P)
    x_all = x_all.reshape(N, T * trial_dim)

    return theta_all, x_all.reshape(N, T * trial_dim)


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
      (1, T*(2+P))
    """
    trial_features = torch.cat([x_o, pulses_o, mask_o], dim=-1)  # (T, 2+P)
    return trial_features.reshape(1, -1).to(torch.float32)
