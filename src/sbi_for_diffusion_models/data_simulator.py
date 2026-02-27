import torch
from torch.distributions import Distribution
from typing import Optional

from sbi_for_diffusion_models.models.rt_choice_model import (
    simulate_rt_choice_batch,
    pack_x_rt_choice,
    generate_pulses_torch,
    max_num_pulses,
    mask_unperceived_pulses,
)
from sbi_for_diffusion_models.run_config import PULSE_INTERVAL


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
    sim_batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate N sessions of T trials each for MNPE training.

    Runs on `device`; outputs are stored on CPU so the full dataset fits in RAM.
    Timeout trials are masked out (mask=0) rather than dropped.

    Returns:
      theta_all: (N, 5) on CPU
      x_all:     (N, T*(2+P+1)) flattened, on CPU
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    trial_dim = 2 + P + 1
    N = int(num_sessions)
    T = int(num_trials)

    theta_all = torch.empty((N, 5), device="cpu", dtype=torch.float32)
    x_all = torch.empty((N, T * trial_dim), device="cpu", dtype=torch.float32)

    if theta is not None:
        theta = theta.to(dtype=torch.float32)
        if theta.ndim == 1:
            if theta.shape[0] != 5:
                raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")
            theta_all.copy_(theta.view(1, 5).expand(N, 5))
        elif theta.ndim == 2:
            if theta.shape != (N, 5):
                raise ValueError(f"theta batch must have shape ({N},5); got {tuple(theta.shape)}")
            theta_all.copy_(theta)
        else:
            raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")
    else:
        theta_all.copy_(prior_theta.sample((N,)).to(dtype=torch.float32))

    B = min(sim_batch_size, N)
    batch_buf = torch.empty((B, T, trial_dim), device=device, dtype=torch.float32)

    for start in range(0, N, sim_batch_size):
        b = min(sim_batch_size, N - start)

        theta_batch = theta_all[start : start + b].to(device=device)  # (b, 5)
        theta_rep   = theta_batch.repeat_interleave(T, dim=0)         # (b*T, 5)

        pulses = generate_pulses_torch(
            n_trials=b * T,
            n_pulses=P,
            p_success=float(p_success),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )  # (b*T, P)

        rt_choice, hit, _ = simulate_rt_choice_batch(
            theta_rep,
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses,
            p_success=float(p_success),
            pulse_generator=gen,
        )  # (b*T, 2), (b*T,)

        hit_2d = hit.view(b, T)              # (b, T)
        mask   = hit_2d.float().unsqueeze(2) # (b, T, 1)

        x_packed = pack_x_rt_choice(rt_choice, log_rt=bool(log_rt)).view(b, T, 2).mul_(mask)

        pulses_masked = mask_unperceived_pulses(
            pulses, rt_choice[:, 0], float(PULSE_INTERVAL)
        ).view(b, T, P)
        pulses_masked = torch.nan_to_num(pulses_masked, nan=0.0)
        pulses_masked.masked_fill_(~hit_2d.unsqueeze(2), 0.0)

        buf = batch_buf[:b]
        buf[:, :, :2      ].copy_(x_packed)
        buf[:, :, 2 : 2+P ].copy_(pulses_masked)
        buf[:, :, 2+P :   ].copy_(mask)
        x_all[start : start + b].copy_(buf.view(b, T * trial_dim))
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
