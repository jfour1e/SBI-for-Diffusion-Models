"""
Tests for pulse generation and tensor utility helpers.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from sbi_for_diffusion_models.models.rt_choice_model import (
    generate_pulses_torch,
    as_pulse_tensor,
    mask_unperceived_pulses,
    max_num_pulses,
)
from sbi_for_diffusion_models.run_config import PULSE_INTERVAL, T_MAX


# max num pulses 
class TestMaxNumPulses:
    def test_consistent_with_config(self):
        expected = int(float(T_MAX) / float(PULSE_INTERVAL))
        assert max_num_pulses() == expected

    def test_positive(self):
        assert max_num_pulses() > 0


# generate_pulses_torch
class TestGeneratePulses:

    @pytest.mark.parametrize("n_trials, n_pulses", [
        (1, 1),
        (1, 40),
        (10, 40),
        (128, 40),
    ])
    def test_shape(self, device, dtype, generator, n_trials, n_pulses):
        s = generate_pulses_torch(
            n_trials=n_trials,
            n_pulses=n_pulses,
            p_success=0.7,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        assert s.shape == (n_trials, n_pulses)

    def test_values_in_plus_minus_one(self, device, dtype, generator):
        s = generate_pulses_torch(
            n_trials=500, n_pulses=40,
            p_success=0.7, device=device, dtype=dtype, generator=generator,
        )
        unique_vals = set(s.unique().tolist())
        assert unique_vals <= {-1.0, 1.0}, f"Unexpected values: {unique_vals}"

    def test_p_success_bias(self, device, dtype):
        """With p_success=1.0 every pulse should match the correct side."""
        g = torch.Generator(device=device)
        g.manual_seed(99)
        s = generate_pulses_torch(
            n_trials=100, n_pulses=40,
            p_success=1.0, device=device, dtype=dtype, generator=g,
        )
        # Within each trial every pulse should be identical (all +1 or all -1)
        for i in range(s.shape[0]):
            assert torch.all(s[i] == s[i, 0]), f"Trial {i} not uniform at p_success=1"

    def test_p_success_half_roughly_balanced(self, device, dtype):
        g = torch.Generator(device=device)
        g.manual_seed(7)
        s = generate_pulses_torch(
            n_trials=2000, n_pulses=40,
            p_success=0.5, device=device, dtype=dtype, generator=g,
        )
        match_frac = (s == s[:, 0:1]).float().mean()
        assert 0.45 < match_frac < 0.55, f"match_frac={match_frac:.3f}"

    def test_empty_trials(self, device, dtype, generator):
        s = generate_pulses_torch(
            n_trials=0, n_pulses=40,
            p_success=0.7, device=device, dtype=dtype, generator=generator,
        )
        assert s.shape == (0, 40)

    def test_empty_pulses(self, device, dtype, generator):
        s = generate_pulses_torch(
            n_trials=5, n_pulses=0,
            p_success=0.7, device=device, dtype=dtype, generator=generator,
        )
        assert s.shape == (5, 0)

    def test_negative_raises(self, device, dtype, generator):
        with pytest.raises(ValueError):
            generate_pulses_torch(
                n_trials=-1, n_pulses=40,
                p_success=0.7, device=device, dtype=dtype, generator=generator,
            )

    def test_dtype_preserved(self, device, generator):
        for dt in [torch.float32, torch.float64]:
            s = generate_pulses_torch(
                n_trials=4, n_pulses=10,
                p_success=0.7, device=device, dtype=dt, generator=generator,
            )
            assert s.dtype == dt


# as_pulse_tensor 
class TestAsPulseTensor:

    def test_from_numpy_2d(self, device, dtype):
        arr = np.array([[1, -1, 1], [-1, 1, -1]], dtype=np.float32)
        t = as_pulse_tensor(arr, device=device, dtype=dtype)
        assert t.shape == (2, 3)
        assert t.device == device

    def test_from_numpy_1d_gets_unsqueezed(self, device, dtype):
        arr = np.array([1, -1, 1], dtype=np.float32)
        t = as_pulse_tensor(arr, device=device, dtype=dtype)
        assert t.shape == (1, 3)

    def test_from_tensor(self, device, dtype):
        t_in = torch.tensor([[1.0, -1.0]])
        t_out = as_pulse_tensor(t_in, device=device, dtype=dtype)
        assert t_out.shape == (1, 2)
        assert t_out.dtype == dtype

    def test_3d_raises(self, device, dtype):
        arr = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            as_pulse_tensor(arr, device=device, dtype=dtype)


# mask_unperceived_pulses ]
class TestMaskUnperceivedPulses:

    def test_basic_masking(self):
        """Pulses after the RT should become NaN."""
        # 2 trials, 4 pulses. Pulses at t=0.25, 0.50, 0.75, 1.00
        pulse_sides = torch.ones(2, 4)
        rt = torch.tensor([0.5, 0.9])  # trial 0 perceives 2, trial 1 perceives 3
        out = mask_unperceived_pulses(pulse_sides, rt, pulse_interval=0.25)

        # Trial 0: pulses at 0.25 and 0.50 perceived, 0.75 and 1.0 masked
        assert not torch.isnan(out[0, 0])
        assert not torch.isnan(out[0, 1])
        assert torch.isnan(out[0, 2])
        assert torch.isnan(out[0, 3])

        # Trial 1: pulses at 0.25, 0.50, 0.75 perceived; 1.0 masked
        assert not torch.isnan(out[1, 2])
        assert torch.isnan(out[1, 3])

    def test_no_masking_when_rt_exceeds_all(self):
        pulse_sides = torch.ones(1, 4)
        rt = torch.tensor([10.0])
        out = mask_unperceived_pulses(pulse_sides, rt, pulse_interval=0.25)
        assert not torch.any(torch.isnan(out))

    def test_all_masked_when_rt_zero(self):
        pulse_sides = torch.ones(1, 4)
        rt = torch.tensor([0.0])
        out = mask_unperceived_pulses(pulse_sides, rt, pulse_interval=0.25)
        assert torch.all(torch.isnan(out))

    def test_does_not_modify_original(self):
        pulse_sides = torch.ones(1, 4)
        rt = torch.tensor([0.3])
        _ = mask_unperceived_pulses(pulse_sides, rt, pulse_interval=0.25)
        assert torch.all(pulse_sides == 1.0), "Original tensor was modified in-place"
