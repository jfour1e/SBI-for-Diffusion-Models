import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.distributions import Distribution

from sbi_for_diffusion_models.models.rt_choice_model import (
    rt_choice_model_simulator_torch,
    simulate_session_data_rt_choice,
    pack_x_rt_choice,
    generate_pulse_matrix_numpy,
    max_num_pulses,
)

def sim_wrapper(
    theta_and_pulses: torch.Tensor, *, mu_sensory: float, p_success: float, P: int, log_rt: bool
) -> torch.Tensor:
    """
    Simulator wrapper that expects concatenated z = [theta(5), pulse_sides(P)].
    Returns packed x = [rt(or log rt), choice].
    """
    theta = theta_and_pulses[:, :5]
    pulse_sides = theta_and_pulses[:, 5 : 5 + P]

    rt_choice = rt_choice_model_simulator_torch(
        theta,
        mu_sensory=mu_sensory,
        pulse_sides=pulse_sides,
        p_success=p_success,  # not used if pulse_sides provided; safe
    )
    return pack_x_rt_choice(rt_choice, log_rt=log_rt)


@torch.no_grad()
def simulate_training_set_with_conditions(
    proposal: Distribution,
    num_simulations: int,
    batch_size: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
    log_rt: bool,
):
    zs = []
    xs = []

    for start in range(0, num_simulations, batch_size):
        bs = min(batch_size, num_simulations - start)
        z = proposal.sample((bs,)).to(device=device, dtype=torch.float32)
        x = sim_wrapper(z, mu_sensory=mu_sensory, p_success=p_success, P=P, log_rt=log_rt)

        zs.append(z.detach().cpu())
        xs.append(x.detach().cpu())

        if (start // batch_size) % 50 == 0:
            print(f"Simulated {start + bs:,}/{num_simulations:,}")

    z_all = torch.cat(zs, dim=0).to(torch.float32)
    x_all = torch.cat(xs, dim=0).to(torch.float32)

    assert z_all.shape[0] == num_simulations
    assert x_all.shape[0] == num_simulations
    assert torch.isfinite(z_all).all()
    assert torch.isfinite(x_all).all()
    assert torch.all((x_all[:, -1] == 0) | (x_all[:, -1] == 1))

    print("Training x shape:", tuple(x_all.shape), " (N,2) = [rt(or log rt), choice]")
    print("Training z shape:", tuple(z_all.shape), " (N, 5+P) = [theta, pulses]")
    print("Unique outcomes in training (choice):", x_all[:, -1].unique().tolist())
    return z_all, x_all


@torch.no_grad()
def simulate_observed_session(
    theta_true: torch.Tensor,
    num_trials: int,
    device,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
    seed: int = 123,
    log_rt: bool,
):
    rng = np.random.default_rng(seed)
    s_np = generate_pulse_matrix_numpy(rng, n_trials=num_trials, n_pulses=P, p_success=p_success)
    pulses_o = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)

    theta_rep = theta_true.view(1, 5).repeat(num_trials, 1)
    rt_choice = rt_choice_model_simulator_torch(
        theta_rep,
        mu_sensory=mu_sensory,
        pulse_sides=pulses_o,
        p_success=p_success,
    )
    x_o = pack_x_rt_choice(rt_choice, log_rt=log_rt)

    return x_o.detach().cpu(), pulses_o.detach().cpu()


def summarize_trials(name: str, x: torch.Tensor) -> None:
    rt = x[:, 0]
    choice = x[:, 1].to(torch.int64)
    counts = torch.bincount(choice, minlength=2)
    frac = counts.float() / counts.sum().clamp_min(1)
    print(
        f"{name}: n={len(x)}  "
        f"rt[min,max]=({rt.min().item():.4f},{rt.max().item():.4f})  "
        f"choice counts={counts.tolist()}  frac={frac.tolist()}"
    )


# ── Session-level simulation for NPE ──────────────────────────────────────────

@torch.no_grad()
def simulate_training_sessions(
    prior_theta,
    num_sessions: int,
    num_trials: int,
    *,
    mu_sensory: float,
    p_success: float,
    P: int,
    log_rt: bool,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate session-level training data for NPE.

    Each session: sample theta, generate pulses, simulate T trials,
    pack each trial as [rt, choice, pulse_1, ..., pulse_P].

    Returns:
        theta_all: (N_sessions, 5)
        x_all:     (N_sessions, T * (2 + P))  flattened session data
    """
    rng = np.random.default_rng(seed)
    trial_dim = 2 + P

    theta_list = []
    x_list = []

    for i in range(num_sessions):
        theta_i = prior_theta.sample((1,)).view(5).to(torch.float32)

        session_seed = int(rng.integers(0, 2**31 - 1))
        session_rng = np.random.default_rng(session_seed)

        pulse_matrix = generate_pulse_matrix_numpy(
            session_rng, n_trials=num_trials, n_pulses=P, p_success=p_success,
        )
        pulse_tensor = torch.from_numpy(pulse_matrix).to(torch.float32)

        x_raw, _ = simulate_session_data_rt_choice(
            theta_i, num_trials, rng=session_rng,
            mu_sensory=mu_sensory,
            pulse_sides=pulse_tensor,
            p_success=p_success,
            return_pulse_sides=True,
        )
        x_packed = pack_x_rt_choice(x_raw, log_rt=log_rt)  # (T, 2)

        trial_features = torch.cat([x_packed, pulse_tensor], dim=-1)  # (T, 2+P)
        x_flat = trial_features.reshape(-1)  # (T*(2+P),)

        theta_list.append(theta_i)
        x_list.append(x_flat)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"Simulated session {i + 1:,}/{num_sessions:,}")

    theta_all = torch.stack(theta_list, dim=0).to(torch.float32)
    x_all = torch.stack(x_list, dim=0).to(torch.float32)

    print(f"theta_train shape: {tuple(theta_all.shape)}")
    print(f"x_train shape: {tuple(x_all.shape)} = (N, T*{trial_dim})")
    return theta_all, x_all


def flatten_observed_session(
    x_o: torch.Tensor,
    pulses_o: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten an observed session for NPE inference.

    Args:
        x_o:      (T, 2)  packed [rt, choice]
        pulses_o: (T, P)

    Returns:
        (1, T*(2+P))  ready for ``posterior.sample(x=...)``
    """
    trial_features = torch.cat([x_o, pulses_o], dim=-1)  # (T, 2+P)
    return trial_features.reshape(1, -1).to(torch.float32)
