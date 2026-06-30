#!/usr/bin/env python
"""Recompute the cross-condition ANOVA and regenerate the group-level plots
from an existing all_animals_all_stages.csv (without re-fitting any animals).

Useful when the stats methodology changes but the per-cell posteriors are
already on disk.

Env vars:
  MODEL_NAME default "lapse_noleak_ar" (determines PARAM_NAMES)
  OUTDIR     default "group_outputs_ar_120k"
  STAGES     default "100-0,90-10,80-20,70-30,60-40,randomProb"
"""
from __future__ import annotations

import os
os.environ.setdefault("MODEL_NAME", "lapse_noleak_ar")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from fit_all_marmosets_all_stages import (
    PARAM_NAMES,
    cross_condition_mixed_anova,
    plot_stage_x_group_lines,
    plot_stage_x_group_violins,
)


def main():
    OUTDIR = os.environ.get("OUTDIR", "group_outputs_ar_120k")
    STAGES = os.environ.get("STAGES", "100-0,90-10,80-20,70-30,60-40,randomProb")
    stages_order = [s.strip() for s in STAGES.split(",") if s.strip()]

    csv_in = os.path.join(OUTDIR, "all_animals_all_stages.csv")
    df = pd.read_csv(csv_in)
    print(f"Loaded {len(df)} rows from {csv_in}")

    tests = cross_condition_mixed_anova(df)
    out_tests = os.path.join(OUTDIR, "cross_condition_tests.csv")
    tests.to_csv(out_tests, index=False)
    print(f"Saved cross-condition ANOVA (cluster-robust): {out_tests}")
    print()
    print(tests.to_string(index=False))
    print()

    out_lines = os.path.join(OUTDIR, "stage_x_group_lines.png")
    plot_stage_x_group_lines(df, stages_order, tests, out_lines)
    print(f"Saved: {out_lines}")

    out_violins = os.path.join(OUTDIR, "stage_x_group_violins.png")
    plot_stage_x_group_violins(df, stages_order, out_violins)
    print(f"Saved: {out_violins}")


if __name__ == "__main__":
    main()
