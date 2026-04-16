"""
Tests for the marmoset behavioral data loader.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from sbi_for_diffusion_models.load_marmoset import (
    _flash_string_to_pulses,
    load_marmoset_sessions,
    P_MAX,
)
from sbi_for_diffusion_models.run_config import PULSE_INTERVAL, T_MAX


# helpers for tests
def _make_synthetic_csv(
    n_sessions: int = 2,
    trials_per_session: int = 100,
    animal: str = "M1",
    stage: str = "70-30",
    seed: int = 0,
) -> str:
    """Write a fake marmoset CSV and return its path."""
    rng = np.random.default_rng(seed)

    import csv as csv_mod

    rng = np.random.default_rng(seed)
    fieldnames = [
        "name", "stage", "session_datetime", "rt",
        "choice", "correct_side", "flashes_left", "flashes_right",
    ]
    rows = []
    for s in range(n_sessions):
        sess_dt = f"2024-01-{10 + s:02d} 10:00:00"
        for t in range(trials_per_session):
            rt = rng.uniform(0.3, 5.0)
            n_pulses = max(1, int(rt / float(PULSE_INTERVAL)))
            # Build mutually-exclusive flash strings (one side per bin)
            fl_chars = []
            fr_chars = []
            for _ in range(n_pulses):
                if rng.random() < 0.5:
                    fl_chars.append("1")
                    fr_chars.append("0")
                else:
                    fl_chars.append("0")
                    fr_chars.append("1")
            fl = "".join(fl_chars)
            fr = "".join(fr_chars)
            correct_side = rng.choice(["left", "right"])
            choice = correct_side if rng.random() < 0.7 else (
                "left" if correct_side == "right" else "right"
            )
            rows.append({
                "name": animal,
                "stage": stage,
                "session_datetime": sess_dt,
                "rt": rt,
                "choice": choice,
                "correct_side": correct_side,
                "flashes_left": fl,
                "flashes_right": fr,
            })

    path = os.path.join(tempfile.mkdtemp(), "marmoset_data.csv")
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, quoting=csv_mod.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return path


# flash_string_to_pulses
class TestFlashStringToPulses:

    def test_basic_right_left(self):
        # 4 bins: right, left, right, left  — RT long enough to see all
        fl = "0101"
        fr = "1010"
        pulses = _flash_string_to_pulses(fl, fr, rt=5.0, pulse_interval=0.25, p_max=10)
        assert pulses.shape == (10,)
        assert pulses[0] == 1.0   # right
        assert pulses[1] == -1.0  # left
        assert pulses[2] == 1.0
        assert pulses[3] == -1.0
        # Remaining should be 0
        assert np.all(pulses[4:] == 0.0)

    def test_masking_by_rt(self):
        # 4 bins but RT only allows 2
        fl = "0101"
        fr = "1010"
        pulses = _flash_string_to_pulses(fl, fr, rt=0.5, pulse_interval=0.25, p_max=10)
        # floor(0.5 / 0.25) = 2, so bins 0 and 1 perceived
        assert pulses[0] == 1.0
        assert pulses[1] == -1.0
        assert pulses[2] == 0.0  # not perceived
        assert pulses[3] == 0.0

    def test_rt_zero_all_masked(self):
        fl = "1111"
        fr = "0000"
        pulses = _flash_string_to_pulses(fl, fr, rt=0.0, pulse_interval=0.25, p_max=5)
        assert np.all(pulses == 0.0)

    def test_output_length_equals_p_max(self):
        fl = "10"
        fr = "01"
        for p_max in [5, 10, 40]:
            pulses = _flash_string_to_pulses(fl, fr, rt=10.0, pulse_interval=0.25, p_max=p_max)
            assert pulses.shape == (p_max,)

    def test_no_flash_in_bin(self):
        # Both "0" → no flash → neutral (0)
        fl = "0000"
        fr = "0000"
        pulses = _flash_string_to_pulses(fl, fr, rt=10.0, pulse_interval=0.25, p_max=5)
        assert np.all(pulses == 0.0)


# load_marmoset_sessions
class TestLoadMarmosetSessions:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.csv_path = _make_synthetic_csv(
            n_sessions=3, trials_per_session=100, animal="M1",
        )
        self.p_max = P_MAX
        self.trial_dim = 2 + self.p_max

    def test_output_types(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80,
        )
        assert isinstance(sessions, list)
        assert all(isinstance(s, torch.Tensor) for s in sessions)
        assert isinstance(meta, list)
        assert all(isinstance(m, dict) for m in meta)

    def test_tensor_shape(self):
        T_per_sess = 80
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=T_per_sess,
        )
        for s, m in zip(sessions, meta):
            T_actual = m["n_trials"]
            assert s.shape == (1, T_actual * self.trial_dim), (
                f"Expected (1, {T_actual * self.trial_dim}), got {s.shape}"
            )

    def test_log_rt_stored(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80, log_rt=True,
        )
        for s, m in zip(sessions, meta):
            T = m["n_trials"]
            x_3d = s.view(T, self.trial_dim)
            rt_col = x_3d[:, 0]
            # log(0.3) ≈ -1.2, log(5.0) ≈ 1.6
            assert torch.all(rt_col < 5.0), "RT values look raw, not log-transformed"

    def test_raw_rt_stored(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80, log_rt=False,
        )
        for s, m in zip(sessions, meta):
            T = m["n_trials"]
            x_3d = s.view(T, self.trial_dim)
            rt_col = x_3d[:, 0]
            assert torch.all(rt_col > 0)

    def test_choice_encoding(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80,
        )
        for s, m in zip(sessions, meta):
            T = m["n_trials"]
            x_3d = s.view(T, self.trial_dim)
            choices = set(x_3d[:, 1].unique().tolist())
            assert choices <= {0.0, 1.0}, f"Unexpected choice values: {choices}"

    def test_pulse_encoding(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80,
        )
        for s, m in zip(sessions, meta):
            T = m["n_trials"]
            x_3d = s.view(T, self.trial_dim)
            pulse_vals = set(x_3d[:, 2:].unique().tolist())
            assert pulse_vals <= {-1.0, 0.0, 1.0}, f"Unexpected pulse values: {pulse_vals}"

    def test_min_trials_filter(self):
        """Sessions with fewer than min_trials should be dropped."""
        csv = _make_synthetic_csv(n_sessions=1, trials_per_session=10, animal="M2")
        sessions, meta = load_marmoset_sessions(
            csv, animal="M2", min_trials=50,
        )
        assert len(sessions) == 0, "Short session should be filtered out"

    def test_subsampling_respects_limit(self):
        csv = _make_synthetic_csv(n_sessions=1, trials_per_session=300, animal="M3")
        T_cap = 100
        sessions, meta = load_marmoset_sessions(
            csv, animal="M3", num_trials_per_session=T_cap,
        )
        assert len(sessions) == 1
        assert meta[0]["n_trials"] == T_cap

    def test_unknown_animal_raises(self):
        with pytest.raises(ValueError, match="No trials found"):
            load_marmoset_sessions(self.csv_path, animal="NONEXISTENT")

    def test_metadata_fields(self):
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1", num_trials_per_session=80,
        )
        for m in meta:
            assert "session_datetime" in m
            assert "n_trials" in m
            assert "accuracy" in m
            assert "rt_median" in m
            assert 0.0 <= m["accuracy"] <= 1.0

    def test_format_compatible_with_npe_embedding(self):
        """
        Verify the flat tensor can be reshaped to (T, 2+P) and that
        each trial's columns align: [rt, choice, pulse_0 .. pulse_{P-1}].
        """
        T_per_sess = 80
        sessions, meta = load_marmoset_sessions(
            self.csv_path, animal="M1",
            num_trials_per_session=T_per_sess, log_rt=True,
        )
        for s, m in zip(sessions, meta):
            T = m["n_trials"]
            x = s.view(T, self.trial_dim)
            # Column 0 = RT (log), Column 1 = choice, Columns 2..end = pulses
            assert x.shape[1] == self.trial_dim
            # Pulse columns should have no NaN (they're already 0-filled)
            assert not torch.any(torch.isnan(x[:, 2:]))