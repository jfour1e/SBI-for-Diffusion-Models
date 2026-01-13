import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.distributions import Distribution

from sbi_for_diffusion_models.models.rt_choice_model import (
    rt_choice_model_simulator_torch, 
    pack_x_rt_choice, 
    generate_pulse_matrix_numpy
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
    assert torch.all((x_all[:, -1] == 0) | (x_all[:, -1] == 1) | (x_all[:, -1] == 2))

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
    counts = torch.bincount(choice, minlength=3)
    frac = counts.float() / counts.sum().clamp_min(1)
    print(
        f"{name}: n={len(x)}  "
        f"rt[min,max]=({rt.min().item():.4f},{rt.max().item():.4f})  "
        f"choice counts={counts.tolist()}  frac={frac.tolist()}"
    )
