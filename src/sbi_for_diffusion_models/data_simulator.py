import torch
from typing import Optional, Tuple 

from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    pack_x_rt_choice,
    generate_pulses_torch,
    max_num_pulses,
    mask_unperceived_pulses,
)
from .run_config import T_MAX, MAX_REGEN, MAX_TIMEOUT_TRIES

@torch.no_grad()
def simulate_training_sessions(
    prior_theta,
    num_sessions: int,
    num_trials: int,
    *,
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
      - generate pulses (T,P) on device
      - simulate T trials with simulate_rt_choice_batch (returns x=(T,2), hit=(T,), s=(T,Pmax))
      - pack x as [rt, choice] (optionally log-rt)
      - concatenate per-trial features as [rt, choice, pulse_1..pulse_P]  (NO mask)
      - flatten to (T*(2+P),)
    """
    N = int(num_sessions)
    T = int(num_trials)
    P = int(P)
    trial_dim = 2 + P

    torch.manual_seed(int(seed)) # torch generator 
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    theta_all = torch.empty((N, 5), device=device, dtype=torch.float32)
    x_all = torch.empty((N, T * trial_dim), device=device, dtype=torch.float32)

    # Optional fixed theta 
    theta_fixed: Optional[torch.Tensor] = None
    if theta is not None:
        theta = theta.to(device=device, dtype=torch.float32)
        if theta.ndim == 1:
            if theta.shape[0] != 5:
                raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")
            theta_fixed = theta.view(1, 5).expand(N, 5)
        elif theta.ndim == 2:
            if theta.shape != (N, 5):
                raise ValueError(f"theta must have shape ({N},5); got {tuple(theta.shape)}")
            theta_fixed = theta
        else:
            raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")

    timeout_frac_allowed = 0.15
    allowed_timeouts = int(torch.ceil(torch.tensor(timeout_frac_allowed * T, device=device)).item())

    for i in range(N):
        # Sample theta or use fixed
        if theta_fixed is None:
            theta_i = prior_theta.sample((1,)).view(5).to(device=device, dtype=torch.float32)
        else:
            theta_i = theta_fixed[i]
        theta_all[i] = theta_i

        # fixed pulses for this session (T,P)
        pulses = generate_pulses_torch(
            n_trials=T,
            n_pulses=P,
            p_success=float(p_success),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        # simulate all trials once
        theta_rep = theta_i.view(1, 5).expand(T, 5)
        x_raw, hit, _ = simulate_rt_choice_batch(
            theta_rep,
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses,
            p_success=float(p_success),
            pulse_generator=gen,
        )

        # retry only timeouts 
        not_hit = ~hit
        regen_used = 0

        for _attempt in range(MAX_TIMEOUT_TRIES):
            idx_all = not_hit.nonzero(as_tuple=False).squeeze(1)
            M_all = int(idx_all.numel())
            if M_all == 0:
                break

            # respect compute budget
            remaining_budget = MAX_REGEN - regen_used
            if remaining_budget <= 0:
                break

            M = min(M_all, int(remaining_budget))
            idx = idx_all[:M]  # deterministic; swap for randperm if you want
            M = int(idx.numel())
            if M == 0:
                break

            theta_sub = theta_i.view(1, 5).expand(M, 5)
            pulses_sub = pulses.index_select(0, idx)

            x_new, hit_new, _ = simulate_rt_choice_batch(
                theta_sub,
                mu_sensory=float(mu_sensory),
                pulse_sides=pulses_sub,
                p_success=float(p_success),
                pulse_generator=gen,
            )

            x_raw.index_copy_(0, idx, x_new)
            hit.index_copy_(0, idx, hit_new)

            regen_used += M
            not_hit = ~hit

        # prior predictive check sent to user -- pass in better prior
        n_timeouts = int(not_hit.sum().item())

        if n_timeouts > allowed_timeouts:
            denom = torch.tensor(max(1, T), device=device, dtype=torch.float32)
            frac = float((torch.tensor(n_timeouts, device=device, dtype=torch.float32) / denom).item())
            raise RuntimeError(
                f"[prior predictive check failed] Too many timeout trials after retries.\n"
                f"Session {i}: timeouts={n_timeouts}/{T} ({frac:.1%}), allowed={allowed_timeouts}/{T} ({timeout_frac_allowed:.0%}).\n"
                f"MAX_TIMEOUT_TRIES={MAX_TIMEOUT_TRIES}, MAX_REGEN={MAX_REGEN}, regen_used={regen_used}.\n"
                f"This likely indicates a bad prior (e.g., drift too small, bounds too large, tau too close to T_MAX, etc.)."
            )
        
        if n_timeouts > 0:
            idx = not_hit.nonzero(as_tuple=False).squeeze(1)
            M = int(idx.numel())

            forced_rt = torch.full((M, 1), float(T_MAX), device=device, dtype=torch.float32)
            forced_choice = torch.randint(0, 2, (M, 1), device=device, dtype=torch.int64).to(torch.float32)
            x_forced = torch.cat([forced_rt, forced_choice], dim=1)

            x_raw.index_copy_(0, idx, x_forced)

        # pack [rt, choice]
        x_packed = pack_x_rt_choice(x_raw, log_rt=bool(log_rt))  # (T,2)

        out = x_all[i].view(T, trial_dim)
        out[:, :2].copy_(x_packed)
        out[:, 2:].copy_(pulses)

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
