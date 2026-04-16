from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .run_config import PULSE_INTERVAL, T_MAX

# Maximum number of pulse bins consistent with current config
P_MAX = int(float(T_MAX) / float(PULSE_INTERVAL))  # 40

def _flash_string_to_pulses(
    flashes_left: str,
    flashes_right: str,
    rt: float,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
) -> np.ndarray:
    """
    Convert a pair of binary flash strings for one trial to a ±1/0 pulse vector.
    """
    n_shown = len(flashes_left)  # string length = number of presented pulses
    n_shown = min(n_shown, p_max)

    # Number of bins the animal could have experienced before responding
    n_perceived = min(math.floor(rt / pulse_interval), n_shown)

    pulses = np.zeros(p_max, dtype=np.float32)

    for k in range(n_perceived):
        left_flash  = flashes_left[k]  == "1"
        right_flash = flashes_right[k] == "1"

        if left_flash:
            side = "left"
        elif right_flash:
            side = "right"
        else:
            continue  # no flash in this bin (treat as 0 / neutral)

        pulses[k] = 1.0 if side == "right" else -1.0

    # bins n_perceived..p_max-1 stay 0 (unperceived / not presented)
    return pulses


def load_marmoset_sessions(
    csv_path: str,
    animal: str,
    stage: str = "70-30",
    num_trials_per_session: int | None = None,
    log_rt: bool = True,
    pulse_interval: float = float(PULSE_INTERVAL),
    p_max: int = P_MAX,
    seed: int = 0,
    min_trials: int = 64,
) -> tuple[list[Tensor], list[dict]]:
    """
    Load and preprocess marmoset behavioral data into NPE-ready session tensors.
    """
    rng = np.random.default_rng(seed)
  
    df = pd.read_csv(
        csv_path,
        compression="infer",
        dtype={"flashes_left": str, "flashes_right": str},
    )
    df = df[(df["name"] == animal) & (df["stage"] == stage)].copy()

    if len(df) == 0:
        raise ValueError(f"No trials found for animal={animal!r}, stage={stage!r}")

    trial_dim = 2 + p_max
    sessions: list[Tensor] = []
    session_meta: list[dict] = []

    for sess_dt, grp in df.groupby("session_datetime"):
        grp = grp.reset_index(drop=True)

        if len(grp) < min_trials:
            continue

        # Subsample if a cap was requested
        if num_trials_per_session is not None and len(grp) > num_trials_per_session:
            idx = rng.choice(len(grp), size=num_trials_per_session, replace=False)
            idx.sort()
            grp = grp.iloc[idx].reset_index(drop=True)

        T = len(grp)
        x = np.zeros((T, trial_dim), dtype=np.float32)

        for i, row in grp.iterrows():
            rt      = float(row["rt"])
            choice  = row["choice"]
            fl      = str(row["flashes_left"])
            fr      = str(row["flashes_right"])

            # RT (log or raw)
            rt_stored = math.log(max(rt, 1e-6)) if log_rt else rt
            # choice in absolute direction (right=1, left=0)
            choice_val = 1.0 if choice == "right" else 0.0

            pulses = _flash_string_to_pulses(
                fl, fr, rt,
                pulse_interval=pulse_interval,
                p_max=p_max,
            )

            x[i, 0]  = rt_stored
            x[i, 1]  = choice_val
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
    print(
        f"Loaded {len(sessions)} sessions for {animal} ({stage})  "
        f"[T_per_session={cap_str}, P={p_max}]"
    )
    for m in session_meta:
        print(
            f"  {m['session_datetime']}  n={m['n_trials']}  "
            f"acc={m['accuracy']:.3f}  median_rt={m['rt_median']:.2f}s"
        )

    return sessions, session_meta