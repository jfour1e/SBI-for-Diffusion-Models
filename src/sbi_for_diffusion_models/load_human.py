"""Load the human pulse-task data into NPE-ready session tensors.

Differences from the marmoset loader:
  * One session per subject (humans do a single session). The CSV has
    `session_datetime` NaN for all human rows; we treat each subject as one
    session by default.
  * The first 10-15 trials per subject are 90/10 training/orientation trials
    (Amanda's email). We filter to `stage_cat == "test"` by default so that
    only the real task trials are used.
  * Per-bin flash encoding: humans have independent left/right Bernoullis per
    bin, so s_k = right - left in {-1, 0, +1} (vs marmoset's exclusive +/-1).
  * Per-trial difficulty (`probflashright`) is heterogeneous within a session
    due to the coin-flip cascade; we preserve that variability in the encoded
    pulse sequence (the network observes the realised pulses, not the assigned
    p_R).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .run_config import PULSE_INTERVAL, T_MAX


P_MAX = int(float(T_MAX) / float(PULSE_INTERVAL))  # 40 bins (same time discretisation)


def _flash_string_to_human_pulses(
    flashes_left: str,
    flashes_right: str,
    rt: float,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
) -> np.ndarray:
    """Encode one human trial as a (p_max,) array of signed pulses.

    s_k = right_flash_k - left_flash_k, in {-1, 0, +1}. Bins beyond the
    presented stimulus length or beyond the subject's RT are set to 0.
    """
    n_shown = min(len(flashes_left), len(flashes_right), p_max)
    n_perceived = min(math.floor(rt / pulse_interval), n_shown)

    pulses = np.zeros(p_max, dtype=np.float32)
    for k in range(n_perceived):
        L = 1 if flashes_left[k] == "1" else 0
        R = 1 if flashes_right[k] == "1" else 0
        pulses[k] = float(R - L)
    return pulses


def load_human_sessions(
    csv_path: str,
    subject: str,
    *,
    log_rt: bool = True,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
    autoregressive: bool = False,
    test_only: bool = True,
    drop_omissions: bool = True,
    min_trials: int = 30,
) -> tuple[list[Tensor], list[dict]]:
    """Load one human subject's behavioural data into NPE-ready session tensors.

    Returns `(sessions, meta)` matching the marmoset loader API.

    sessions: list of 1-by-(T*trial_dim) float32 tensors, one per session.
              In the human task there is one session per subject.
    meta:     parallel list of dicts with summary stats.

    With `autoregressive=True`, per-trial features are
    [log_rt, choice, prev_choice_signed, prev_outcome_signed, s_1..p_max],
    so trial_dim = 4 + p_max. Without AR, trial_dim = 2 + p_max.
    """
    df = pd.read_csv(
        csv_path,
        compression="infer",
        dtype={"flashes_left": str, "flashes_right": str},
        low_memory=False,
    )
    df = df[(df["specie"] == "Human") & (df["name"] == subject)].copy()
    if len(df) == 0:
        raise ValueError(f"No human trials found for subject={subject!r}")

    if test_only:
        df = df[df["stage_cat"] == "test"]
    if drop_omissions:
        df = df[df["outcome"] != "omission"]
    df = df.dropna(subset=["rt", "choice", "correct_side", "flashes_left", "flashes_right"])
    # Preserve presentation order using the `trial` column
    df = df.sort_values("trial").reset_index(drop=True)

    if len(df) < min_trials:
        cap_str = "all"
        return [], []

    trial_dim = (4 if autoregressive else 2) + p_max
    T = len(df)
    x = np.zeros((T, trial_dim), dtype=np.float32)

    prev_choice_signed = 0.0
    prev_outcome_signed = 0.0

    for i, row in df.iterrows():
        rt = float(row["rt"])
        choice = row["choice"]
        fl = str(row["flashes_left"])
        fr = str(row["flashes_right"])

        rt_stored = math.log(max(rt, 1e-6)) if log_rt else rt
        choice_val = 1.0 if choice == "right" else 0.0

        pulses = _flash_string_to_human_pulses(fl, fr, rt, pulse_interval, p_max)

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

    x_flat = torch.from_numpy(x).reshape(1, -1)
    acc = float((df["choice"] == df["correct_side"]).mean())
    meta = [{
        "subject": subject,
        "n_trials": T,
        "accuracy": acc,
        "rt_median": float(df["rt"].median()),
        "stage_cat": "test" if test_only else "all",
    }]
    print(
        f"Loaded {T} test trials for human {subject}  "
        f"acc={acc:.3f}  median_rt={df['rt'].median():.2f}s  "
        f"{'AR' if autoregressive else 'non-AR'}  trial_dim={trial_dim}"
    )
    return [x_flat], meta


def load_human_sessions_compare(
    csv_path: str,
    animal: str,
    stage: str = "test",
    *,
    log_rt: bool = True,
    seed: int = 0,
    autoregressive: bool = False,
):
    """Uniform-signature adapter for scripts/model_comparison.py.

    Humans do a single 'test' session per subject, so `stage` and `seed` are
    accepted (for a common loader API across species) but unused. `animal` is the
    subject name.
    """
    return load_human_sessions(
        csv_path, subject=animal, log_rt=log_rt, autoregressive=autoregressive,
    )
