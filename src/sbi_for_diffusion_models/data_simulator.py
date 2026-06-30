import torch
from collections.abc import Sequence
from typing import Optional, Callable
import math
import os
import warnings
import glob

from sbi_for_diffusion_models.models.rt_choice_model import (
    pack_x_rt_choice,
    generate_pulses_torch,
    mask_unperceived_pulses,
)
from .run_config import T_MAX, PULSE_INTERVAL, MAX_TIMEOUT_TRIES, TIMEOUT_FRAC_ALLOWED


def trial_feature_dim(P: int, *, ar: bool = False) -> int:
    """Per-trial feature length.

    iid (ar=False): [log_rt, choice, pulse_1..P]                      -> 2 + P
    AR  (ar=True):  [log_rt, choice, prev_choice, prev_outcome, p_1..P] -> 4 + P
    """
    return (4 if ar else 2) + int(P)


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
    pulse_generator_fn: Optional[Callable] = None,
    p_success_per_trial_fn: Optional[Callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-autoregressive session simulator.

    Marmoset/mouse path (defaults): per-session p_success constant across trials,
    pulses from `generate_pulses_torch` (XOR-style Bernoulli).

    Human path (pass `pulse_generator_fn=generate_pulses_human_torch` and
    `p_success_per_trial_fn=sample_p_success_human_cascade`): every trial gets a
    fresh cascade-sampled p_R and two independent per-side Bernoulli pulses. The
    non-AR path has no sequential dependency, so all N*T trials are sampled at
    once.
    """
    N = int(num_sessions)
    T = int(num_trials)
    P = int(P)
    trial_dim = trial_feature_dim(P)
    NT = N * T

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    _pulse_gen = pulse_generator_fn if pulse_generator_fn is not None else generate_pulses_torch

    p_success_session = _p_success_by_session(
        p_success,
        num_sessions=N,
        device=device,
        generator=gen,
    ).clamp(0.0, 1.0)
    if p_success_per_trial_fn is not None:
        # Fresh per-trial p_R for every one of the N*T trials (human cascade).
        p_success_trial = p_success_per_trial_fn(NT, device=device, generator=gen)
    else:
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

    pulses = _pulse_gen(
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
        pulses_sub = _pulse_gen(
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


@torch.no_grad()
def simulate_training_sessions_ar(
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
    pulse_generator_fn: Optional[Callable] = None,
    p_success_per_trial_fn: Optional[Callable] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive variant: trials within a session are simulated sequentially
    so the previous choice/outcome can feed into the next trial.

    `simulate_batch_fn` must accept `prev_choice_signed` and `prev_outcome_signed`
    kwargs (e.g. `simulate_rt_choice_batch_ar`).

    Per-trial features stored as: [log_rt, choice, prev_choice_signed,
    prev_outcome_signed, pulse_1..P], so trial_dim = 4 + P.

    Marmoset path (defaults): per-session p_success is constant across trials,
    and pulses are generated with `generate_pulses_torch` (XOR-style Bernoulli).

    Human path (pass `pulse_generator_fn=generate_pulses_human_torch` and
    `p_success_per_trial_fn=sample_p_success_human_cascade`): each trial gets
    a fresh p_R via the cascade, and pulses are two independent Bernoullis
    per side per bin.
    """
    N = int(num_sessions)
    T = int(num_trials)
    P = int(P)
    trial_dim = trial_feature_dim(P, ar=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    p_success_session = _p_success_by_session(
        p_success,
        num_sessions=N,
        device=device,
        generator=gen,
    ).clamp(0.0, 1.0)

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
            raise ValueError(f"theta must be 1-D or 2-D; got ndim={theta_all.ndim}")
    else:
        theta_all = torch.as_tensor(
            prior_theta.sample((N,)),
            device=device,
            dtype=torch.float32,
        )
        if theta_all.ndim == 1:
            theta_all = theta_all.unsqueeze(-1)

    allowed_timeouts = math.ceil(TIMEOUT_FRAC_ALLOWED * T)
    timeouts_per_session = torch.zeros((N,), device=device, dtype=torch.int64)

    prev_choice_signed = torch.zeros((N,), device=device, dtype=torch.float32)
    prev_outcome_signed = torch.zeros((N,), device=device, dtype=torch.float32)

    x_all = torch.empty((N, T, trial_dim), device=device, dtype=torch.float32)

    _pulse_gen = pulse_generator_fn if pulse_generator_fn is not None else generate_pulses_torch

    for t in range(T):
        # Per-trial p_R: cascade-sampled for human, constant per session for marmoset
        if p_success_per_trial_fn is not None:
            p_for_trial = p_success_per_trial_fn(N, device=device, generator=gen)
        else:
            p_for_trial = p_success_session

        pulses, correct_side = _pulse_gen(
            n_trials=N,
            n_pulses=P,
            p_success=p_for_trial,
            device=device,
            dtype=torch.float32,
            generator=gen,
            return_correct_side=True,
        )

        x_raw, hit, _ = simulate_batch_fn(
            theta_all,
            mu_sensory=float(mu_sensory),
            pulse_sides=pulses,
            p_success=p_for_trial,
            pulse_generator=gen,
            prev_choice_signed=prev_choice_signed,
            prev_outcome_signed=prev_outcome_signed,
        )

        for _ in range(int(MAX_TIMEOUT_TRIES)):
            idx = (~hit).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            M = idx.numel()
            # Resample fresh p_R + pulses for the retry subset (same scheme as primary path)
            if p_success_per_trial_fn is not None:
                p_sub = p_success_per_trial_fn(M, device=device, generator=gen)
            else:
                p_sub = p_for_trial[idx] if isinstance(p_for_trial, torch.Tensor) else p_for_trial
            pulses_sub, correct_side_sub = _pulse_gen(
                n_trials=M,
                n_pulses=P,
                p_success=p_sub,
                device=device,
                dtype=torch.float32,
                generator=gen,
                return_correct_side=True,
            )
            x_new, hit_new, _ = simulate_batch_fn(
                theta_all[idx],
                mu_sensory=float(mu_sensory),
                pulse_sides=pulses_sub,
                p_success=p_sub,
                pulse_generator=gen,
                prev_choice_signed=prev_choice_signed[idx],
                prev_outcome_signed=prev_outcome_signed[idx],
            )
            x_raw.index_copy_(0, idx, x_new)
            hit.index_copy_(0, idx, hit_new)
            pulses.index_copy_(0, idx, pulses_sub)
            correct_side.index_copy_(0, idx, correct_side_sub)

        not_hit = ~hit
        timeouts_per_session = timeouts_per_session + not_hit.to(torch.int64)
        idx = not_hit.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            x_raw[idx, 0] = float(T_MAX)
            x_raw[idx, 1] = torch.randint(
                0, 2, (idx.numel(),), device=device, generator=gen
            ).to(x_raw.dtype)

        rt_raw = x_raw[:, 0]
        pulses_masked = mask_unperceived_pulses(pulses, rt_raw, float(PULSE_INTERVAL))
        pulses_masked = torch.nan_to_num(pulses_masked, nan=0.0)

        x_packed = pack_x_rt_choice(x_raw, log_rt=bool(log_rt))  # (N, 2)

        x_all[:, t, 0:2] = x_packed
        x_all[:, t, 2] = prev_choice_signed
        x_all[:, t, 3] = prev_outcome_signed
        x_all[:, t, 4:] = pulses_masked

        chose_right = x_raw[:, 1] >= 0.5
        next_choice_signed = torch.where(
            chose_right,
            torch.ones_like(prev_choice_signed),
            -torch.ones_like(prev_choice_signed),
        )
        correct = chose_right == (correct_side > 0)
        next_outcome_signed = torch.where(
            correct,
            torch.ones_like(prev_outcome_signed),
            -torch.ones_like(prev_outcome_signed),
        )
        prev_choice_signed = next_choice_signed
        prev_outcome_signed = next_outcome_signed

    if warn_on_timeouts:
        n_bad = int((timeouts_per_session > allowed_timeouts).sum().item())
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} sessions exceeded timeout threshold after retries (AR).",
                RuntimeWarning,
                stacklevel=2,
            )

    return theta_all, x_all.reshape(N, T * trial_dim)


# ---------------------------------------------------------------------------
# Pre-simulated corpus IO
#
# Pre-simulating a large (theta, x) corpus to disk lets us decouple expensive
# AR simulation from training: an SCC array job fans out chunks in parallel,
# then training is a near-free DataLoader-style sample-from-RAM loop.
#
# On-disk layout:
#   <root>/<model_name>/chunk_<idx:04d>.pt   # {'theta': (N,D), 'x': (N,Xdim), 'cfg': RunConfig}
#   <root>/<model_name>/val_<idx:04d>.pt     # same shape; identified by filename
#
# Train chunks are sampled with replacement; val chunks are concatenated and
# used as the held-out set.
# ---------------------------------------------------------------------------


def corpus_dir(root: str, model_name: str) -> str:
    return os.path.join(root, model_name)


def chunk_filename(idx: int, *, kind: str = "train") -> str:
    """Filename for chunk `idx`; `kind` in {'train', 'val'}."""
    prefix = "chunk" if kind == "train" else "val"
    return f"{prefix}_{int(idx):04d}.pt"


@torch.no_grad()
def simulate_chunk_to_disk(
    out_path: str,
    *,
    prior_theta,
    num_sessions: int,
    num_trials: int,
    simulate_batch_fn: Callable,
    device: torch.device,
    mu_sensory: float,
    p_success,
    P: int,
    log_rt: bool,
    seed: int,
    autoregressive: bool,
    cfg,
    pulse_generator_fn: Optional[Callable] = None,
    p_success_per_trial_fn: Optional[Callable] = None,
) -> tuple[int, int]:
    """Simulate one chunk and save it to disk. Returns (theta_dim, x_dim)."""
    session_fn = simulate_training_sessions_ar if autoregressive else simulate_training_sessions
    extra = {}
    if pulse_generator_fn is not None or p_success_per_trial_fn is not None:
        # Supported in both AR and non-AR session sims (human pulse generator +
        # per-trial p_R cascade).
        extra["pulse_generator_fn"] = pulse_generator_fn
        extra["p_success_per_trial_fn"] = p_success_per_trial_fn
    theta, x = session_fn(
        prior_theta,
        num_sessions=int(num_sessions),
        num_trials=int(num_trials),
        simulate_batch_fn=simulate_batch_fn,
        device=device,
        mu_sensory=float(mu_sensory),
        p_success=p_success,
        P=int(P),
        log_rt=bool(log_rt),
        seed=int(seed),
        warn_on_timeouts=False,
        **extra,
    )
    theta = theta.detach().to("cpu", dtype=torch.float32).contiguous()
    x = x.detach().to("cpu", dtype=torch.float32).contiguous()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "theta": theta,
            "x": x,
            "cfg": cfg,
            "autoregressive": bool(autoregressive),
            "seed": int(seed),
            "num_sessions": int(num_sessions),
            "num_trials": int(num_trials),
            "P": int(P),
        },
        out_path,
    )
    return int(theta.shape[-1]), int(x.shape[-1])


def _validate_chunks_match(chunks: list[dict], expected_ar: bool) -> None:
    if not chunks:
        return
    ref = chunks[0]
    for c in chunks[1:]:
        if c["theta"].shape[-1] != ref["theta"].shape[-1]:
            raise RuntimeError(
                f"Corpus chunk theta_dim mismatch: {c['theta'].shape[-1]} vs {ref['theta'].shape[-1]}"
            )
        if c["x"].shape[-1] != ref["x"].shape[-1]:
            raise RuntimeError(
                f"Corpus chunk x_dim mismatch: {c['x'].shape[-1]} vs {ref['x'].shape[-1]}"
            )
    for c in chunks:
        if bool(c.get("autoregressive", False)) != bool(expected_ar):
            raise RuntimeError(
                f"Corpus chunk autoregressive={c.get('autoregressive')!r} "
                f"does not match expected ar={expected_ar}"
            )


def load_corpus(
    dirpath: str,
    *,
    autoregressive: bool,
    pattern: str = "chunk_*.pt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all chunks matching `pattern` from `dirpath` into one (theta, x) pair.

    Returns concatenated (theta, x) tensors on CPU. Caller moves to device.
    """
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        raise FileNotFoundError(f"No corpus chunks matching {pattern!r} in {dirpath}")
    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    _validate_chunks_match(chunks, expected_ar=autoregressive)
    theta = torch.cat([c["theta"] for c in chunks], dim=0)
    x = torch.cat([c["x"] for c in chunks], dim=0)
    print(
        f"[corpus] loaded {len(files)} chunk(s) from {dirpath} "
        f"-> theta {tuple(theta.shape)}, x {tuple(x.shape)} (ar={autoregressive})"
    )
    return theta, x
