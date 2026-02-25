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
    N = int(num_sessions)
    T = int(num_trials)

    theta_all = torch.empty((N, 5), device=device, dtype=torch.float32)
    x_all = torch.empty((N, T * trial_dim), device=device, dtype=torch.float32)

    x_pad = torch.zeros((T, 2), device=device, dtype=torch.float32)
    p_pad = torch.zeros((T, P), device=device, dtype=torch.float32)
    mask_pad = torch.zeros((T, 1), device=device, dtype=torch.float32)

    # --- NEW: normalize theta input (if provided) ---
    theta_fixed = None
    if theta is not None:
        theta = theta.to(device=device, dtype=torch.float32)
        if theta.ndim == 1:
            if theta.shape[0] != 5:
                raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")
            theta_fixed = theta.view(1, 5).expand(N, 5)
        elif theta.ndim == 2:
            if theta.shape != (N, 5):
                raise ValueError(f"theta batch must have shape ({N},5); got {tuple(theta.shape)}")
            theta_fixed = theta
        else:
            raise ValueError(f"theta must have shape (5,) or (N,5); got {tuple(theta.shape)}")

    for i in range(N):
        theta_i = prior_theta.sample((1,)).view(5).to(device=device, dtype=torch.float32) if theta_fixed is None else theta_fixed[i]
        theta_all[i] = theta_i

        # pulses: (T,P)
        pulses = generate_pulses_torch(
            n_trials=T,
            n_pulses=P,
            p_success=float(p_success),
            device=device,
            dtype=torch.float32,
            generator=gen,
        )

        # Expand without allocating (repeat() allocates)
        theta_rep = theta_i.view(1, 5).expand(T, 5)

        rt_choice, hit, _ = simulate_rt_choice_batch(
            theta_rep,
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses,
            p_success=float(p_success),
            pulse_generator=gen,
        )
        x_packed = pack_x_rt_choice(rt_choice, log_rt=bool(log_rt))  # (T,2)

        # --- Drop timeouts + pad back to T + mask ---
        # Fill buffers with zeros (cheap) then overwrite first n_keep
        x_keep = x_pad  # view alias (we'll overwrite slices)
        p_keep = p_pad
        mask = mask_pad

        # Indices of hits (same as before)
        keep_idx = hit.nonzero(as_tuple=False).squeeze(1)
        n_keep = int(keep_idx.numel())

        if n_keep > 0:
            if n_keep >= T:
                keep_idx = keep_idx[:T]
                n_keep = T
            # overwrite first n_keep entries (rest remain zeros)
            x_keep[:n_keep].copy_(x_packed.index_select(0, keep_idx))
            p_keep[:n_keep].copy_(pulses.index_select(0, keep_idx))
            mask[:n_keep].fill_(1.0)

            if n_keep < T:
                # Ensure remaining mask is 0 (might be left 1 from previous iter)
                mask[n_keep:].zero_()
        else:
            # ensure mask all zeros
            mask.zero_()

        # Write directly into flat output (no torch.cat)
        # Layout: [rt, choice, pulses..., mask]
        out = x_all[i].view(T, trial_dim)
        out[:, :2].copy_(x_keep)
        out[:, 2 : 2 + P].copy_(p_keep)
        out[:, 2 + P :].copy_(mask)

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