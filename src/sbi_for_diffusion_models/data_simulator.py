import torch
from collections.abc import Sequence
from typing import Optional, Callable
import math
import warnings

from sbi_for_diffusion_models.models.rt_choice_model import (
    pack_x_rt_choice,
    generate_pulses_torch,
    mask_unperceived_pulses,
)
from .run_config import T_MAX, PULSE_INTERVAL, MAX_TIMEOUT_TRIES, TIMEOUT_FRAC_ALLOWED


def trial_feature_dim(P: int) -> int:
    return 2 + int(P)


def _p_success_by_session(
    p_success: float | Sequence[float] | torch.Tensor,
    *,
    num_sessions: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if isinstance(p_success, torch.Tensor):
        values = p_success.to(device=device, dtype=torch.float32).flatten()
    elif isinstance(p_success, (str, bytes)):
        raise TypeError("p_success must be numeric, not a string")
    elif isinstance(p_success, Sequence):
        values = torch.tensor(list(p_success), device=device, dtype=torch.float32).flatten()
    else:
        return torch.full((num_sessions,), float(p_success), device=device, dtype=torch.float32)

    if values.numel() == 0:
        raise ValueError("p_success candidate list must not be empty")
    if values.numel() == 1:
        return values.expand(num_sessions).contiguous()
    if values.numel() == num_sessions:
        return values.contiguous()

    idx = torch.randint(
        low=0,
        high=int(values.numel()),
        size=(num_sessions,),
        device=device,
        generator=generator,
    )
    return values[idx].contiguous()


@torch.no_grad()
def simulate_training_sessions(
    prior_theta,
    num_sessions: int,
    num_trials: int,
    *,
    simulate_batch_fn: Callable,
    device: torch.device,
    mu_sensory: float,
    p_success: float | Sequence[float] | torch.Tensor,
    P: int,
    log_rt: bool,
    seed: int = 0,
    theta: Optional[torch.Tensor] = None,
    warn_on_timeouts: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    N = int(num_sessions)
    T = int(num_trials)
    P = int(P)
    trial_dim = trial_feature_dim(P)
    NT = N * T

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    p_success_session = _p_success_by_session(
        p_success,
        num_sessions=N,
        device=device,
        generator=gen,
    ).clamp(0.0, 1.0)
    p_success_trial = (
        p_success_session[:, None]
        .expand(N, T)
        .reshape(NT)
        .contiguous()
    )

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
        theta_all = torch.as_tensor(
            prior_theta.sample((N,)),
            device=device,
            dtype=torch.float32,
        )
        if theta_all.ndim == 1:
            theta_all = theta_all.unsqueeze(-1)

    theta_batch = theta_all[:, None, :].expand(N, T, theta_all.shape[1]).reshape(NT, -1)

    pulses = generate_pulses_torch(
        n_trials=NT,
        n_pulses=P,
        p_success=p_success_trial,
        device=device,
        dtype=torch.float32,
        generator=gen,
    )

    x_raw, hit, _ = simulate_batch_fn(
        theta_batch,
        mu_sensory=float(mu_sensory),
        pulse_sides=pulses,
        p_success=p_success_trial,
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
            p_success=p_success_trial[idx],
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        x_new, hit_new, _ = simulate_batch_fn(
            theta_batch[idx],
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses_sub,
            p_success=p_success_trial[idx],
            pulse_generator=gen,
        )

        x_raw.index_copy_(0, idx, x_new)
        hit.index_copy_(0, idx, hit_new)
        pulses.index_copy_(0, idx, pulses_sub)

    not_hit = ~hit
    allowed_timeouts = math.ceil(TIMEOUT_FRAC_ALLOWED * T)
    timeouts_per_session = not_hit.view(N, T).sum(dim=1)

    if warn_on_timeouts:
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

    rt_raw = x_raw[:, 0]
    pulses = mask_unperceived_pulses(pulses, rt_raw, float(PULSE_INTERVAL))
    pulses = torch.nan_to_num(pulses, nan=0.0)

    x_packed = pack_x_rt_choice(x_raw, log_rt=bool(log_rt))

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
    _ = mask_o
    trial_features = torch.cat([x_o, pulses_o], dim=-1)
    return trial_features.reshape(1, -1).to(torch.float32)
