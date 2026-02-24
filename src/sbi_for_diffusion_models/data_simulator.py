import torch
from torch.distributions import Distribution
from typing import Optional

from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    pack_x_rt_choice,
    generate_pulses_torch,
    max_num_pulses,
)

# ── Session-level simulation for NPE ──────────────────────────────────────────
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
    theta: Optional[torch.Tensor] = None,   # NEW
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MNPE session simulation with timeout removal via mask-aware padding.

    Returns:
      theta_all: (N_sessions, 5) on `device`
      x_all:     (N_sessions, T*(2+P+1)) flattened, with per-trial mask appended.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    trial_dim = 2 + P + 1
    theta_all = torch.empty((num_sessions, 5), device=device, dtype=torch.float32)
    x_all = torch.empty((num_sessions, num_trials * trial_dim), device=device, dtype=torch.float32)

    # --- NEW: normalize theta input (if provided) ---
    theta_fixed = None
    if theta is not None:
        theta = theta.to(device=device, dtype=torch.float32)
        if theta.ndim == 1:
            if theta.shape[0] != 5:
                raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")
            theta_fixed = theta.view(1, 5).expand(num_sessions, 5)
        elif theta.ndim == 2:
            if theta.shape != (num_sessions, 5):
                raise ValueError(
                    f"theta batch must have shape ({num_sessions},5); got {tuple(theta.shape)}"
                )
            theta_fixed = theta
        else:
            raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")

    for i in range(num_sessions):
        if theta_fixed is None:
            theta_i = prior_theta.sample((1,)).view(5).to(device=device, dtype=torch.float32)
        else:
            theta_i = theta_fixed[i]

        theta_all[i] = theta_i

        pulses = generate_pulses_torch(
            n_trials=num_trials,
            n_pulses=P,
            p_success=float(p_success),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        theta_rep = theta_i.view(1, 5).repeat(num_trials, 1)
        rt_choice, hit, _ = simulate_rt_choice_batch(
            theta_rep,
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses,
            p_success=float(p_success),
            pulse_generator=gen,
        )
        x_packed = pack_x_rt_choice(rt_choice, log_rt=bool(log_rt))

        # drop timeouts, pad, mask (your existing code unchanged)
        keep_idx = torch.where(hit)[0]
        x_keep = x_packed[keep_idx]
        p_keep = pulses[keep_idx]

        n_keep = x_keep.shape[0]
        if n_keep >= num_trials:
            x_keep = x_keep[:num_trials]
            p_keep = p_keep[:num_trials]
            mask = torch.ones((num_trials, 1), device=device, dtype=torch.float32)
        else:
            pad_n = num_trials - n_keep
            x_pad = torch.zeros((pad_n, 2), device=device, dtype=torch.float32)
            p_pad = torch.zeros((pad_n, P), device=device, dtype=torch.float32)
            mask = torch.cat(
                [
                    torch.ones((n_keep, 1), device=device, dtype=torch.float32),
                    torch.zeros((pad_n, 1), device=device, dtype=torch.float32),
                ],
                dim=0,
            )
            x_keep = torch.cat([x_keep, x_pad], dim=0)
            p_keep = torch.cat([p_keep, p_pad], dim=0)

        trial_features = torch.cat([x_keep, p_keep, mask], dim=-1)
        x_all[i] = trial_features.reshape(-1)

    return theta_all, x_all

def flatten_observed_session(
    x_o: torch.Tensor,
    pulses_o: torch.Tensor,
    mask_o: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten observed session for NPE inference, including mask.

    Args:
      x_o: (T,2) packed [rt, choice] on correct device
      pulses_o: (T,P)
      mask_o: (T,1) float in {0,1}

    Returns:
      (1, T*(2+P+1))
    """
    trial_features = torch.cat([x_o, pulses_o, mask_o], dim=-1)  # (T, 2+P+1)
    return trial_features.reshape(1, -1).to(torch.float32)