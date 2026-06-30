"""Load the mouse pulse-task data into NPE-ready session tensors.

Mouse task layout (from ALLspecies_trials_combined.csv, specie == "Mouse"):
  * Pulse schedule is 10 Hz: PULSE_INTERVAL = 0.10 s, T_MAX = 5 s -> P = 50 bins.
    Set SPECIES=mouse (run_config) so the module globals match.
  * Difficulty is given by `stage_cat`: "100-0" (p=1.0), "90-10" (p=0.9),
    "80-20" (p=0.8). The numeric `stage` column is 3/4/5 respectively.
    `probflashright` is NaN for mice, so p_success is taken from stage_cat.
  * Flash encoding is EXCLUSIVE, like the marmoset: exactly one side flashes
    per bin (never both, never neither), so s_k in {-1, +1}. This differs from
    the human task (independent Bernoullis, s_k in {-1, 0, +1}).
  * Flash strings are RESPONSE-TERMINATED: len(flashes_left) == floor(rt/0.1)
    == the number of pulses the animal actually experienced before responding.
    There are no recorded pulses beyond the response, and no omissions/timeouts
    (every trial has a left/right choice with rt in [0.1, 5.0]).
  * Multiple sessions per animal (`session_datetime`); `correct_side` is present
    for autoregressive win-stay/lose-shift threading.

API matches load_marmoset_sessions / load_human_sessions.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .run_config import PULSE_INTERVAL, T_MAX

# Maximum number of pulse bins consistent with the current (mouse) config.
P_MAX = int(round(float(T_MAX) / float(PULSE_INTERVAL)))  # 50 when SPECIES=mouse

# stage_cat -> intended p_success (probability a pulse points to the correct side)
STAGE_CAT_P_SUCCESS = {"100-0": 1.0, "90-10": 0.9, "80-20": 0.8}


def _flash_string_to_pulses(
    flashes_left: str,
    flashes_right: str,
    rt: float,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
) -> np.ndarray:
    """Encode one mouse trial as a (p_max,) signed-pulse vector in {-1, 0, +1}.

    Exclusive scheme: s_k = +1 if the right side flashed in bin k, -1 if the
    left side flashed, 0 if neither (does not occur in real mouse data, but the
    branch is kept for robustness). Strings are already response-terminated, so
    n_perceived == string length; we still clamp by floor(rt/interval) and p_max
    defensively.
    """
    n_shown = min(len(flashes_left), len(flashes_right), p_max)
    n_perceived = min(int(math.floor(rt / pulse_interval)), n_shown)

    pulses = np.zeros(p_max, dtype=np.float32)
    for k in range(n_perceived):
        left_flash = flashes_left[k] == "1"
        right_flash = flashes_right[k] == "1"
        if right_flash and not left_flash:
            pulses[k] = 1.0
        elif left_flash and not right_flash:
            pulses[k] = -1.0
        # both or neither -> leave 0 (not expected in mouse data)
    return pulses


def load_mouse_sessions(
    csv_path: str,
    animal: str,
    stage: str = "80-20",
    *,
    num_trials_per_session: int | None = None,
    log_rt: bool = True,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
    seed: int = 0,
    min_trials: int = 64,
    autoregressive: bool = False,
) -> tuple[list[Tensor], list[dict]]:
    """Load one mouse's behavioural data at one stage into NPE-ready sessions.

    `stage` is a stage_cat string ("100-0" | "90-10" | "80-20").

    Returns `(sessions, meta)`:
      sessions: list of 1-by-(T*trial_dim) float32 tensors, one per session.
      meta:     parallel list of per-session summary dicts.

    With `autoregressive=True`, per-trial features are
    [log_rt, choice, prev_choice_signed, prev_outcome_signed, s_1..p_max]
    (trial_dim = 4 + P); otherwise [log_rt, choice, s_1..p_max] (trial_dim = 2 + P).
    Trial order within a session is preserved (sorted by trial_datetime if
    present, else by `trial`, else row order) before AR threading.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(
        csv_path,
        compression="infer",
        dtype={"flashes_left": str, "flashes_right": str},
        low_memory=False,
    )
    df = df[(df["specie"] == "Mouse")
            & (df["name"] == animal)
            & (df["stage_cat"] == stage)].copy()

    if len(df) == 0:
        raise ValueError(
            f"No mouse trials for animal={animal!r}, stage_cat={stage!r}"
        )

    df = df.dropna(subset=["rt", "choice", "correct_side",
                           "flashes_left", "flashes_right"])

    trial_dim = (4 if autoregressive else 2) + p_max
    sessions: list[Tensor] = []
    session_meta: list[dict] = []

    sort_key = "trial_datetime" if "trial_datetime" in df.columns else "trial"

    for sess_dt, grp in df.groupby("session_datetime"):
        if sort_key in grp.columns:
            grp = grp.sort_values(sort_key)
        grp = grp.reset_index(drop=True)

        if len(grp) < min_trials:
            continue

        if num_trials_per_session is not None and len(grp) > num_trials_per_session:
            idx = rng.choice(len(grp), size=num_trials_per_session, replace=False)
            idx.sort()
            grp = grp.iloc[idx].reset_index(drop=True)

        T = len(grp)
        x = np.zeros((T, trial_dim), dtype=np.float32)

        prev_choice_signed = 0.0
        prev_outcome_signed = 0.0

        for i, row in grp.iterrows():
            rt = float(row["rt"])
            choice = row["choice"]
            fl = str(row["flashes_left"])
            fr = str(row["flashes_right"])

            rt_stored = math.log(max(rt, 1e-6)) if log_rt else rt
            choice_val = 1.0 if choice == "right" else 0.0

            pulses = _flash_string_to_pulses(
                fl, fr, rt, pulse_interval=pulse_interval, p_max=p_max,
            )

            x[i, 0] = rt_stored
            x[i, 1] = choice_val
            if autoregressive:
                x[i, 2] = prev_choice_signed
                x[i, 3] = prev_outcome_signed
                x[i, 4:] = pulses

                this_choice_signed = 1.0 if choice == "right" else -1.0
                correct = (choice == row["correct_side"])
                this_outcome_signed = 1.0 if correct else -1.0
                prev_choice_signed = this_choice_signed
                prev_outcome_signed = this_outcome_signed
            else:
                x[i, 2:] = pulses

        x_flat = torch.from_numpy(x).reshape(1, -1)  # (1, T * trial_dim)

        acc = float((grp["choice"] == grp["correct_side"]).mean())
        sessions.append(x_flat)
        session_meta.append({
            "session_datetime": sess_dt,
            "n_trials": T,
            "accuracy": acc,
            "rt_median": float(grp["rt"].median()),
        })

    cap_str = str(num_trials_per_session) if num_trials_per_session is not None else "all"
    ar_str = " AR" if autoregressive else ""
    print(
        f"Loaded {len(sessions)} sessions for mouse {animal} ({stage}){ar_str}  "
        f"[T_per_session={cap_str}, P={p_max}, trial_dim={trial_dim}]"
    )
    return sessions, session_meta
