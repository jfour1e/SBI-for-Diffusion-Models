#!/usr/bin/env python
"""Merge per-animal model-comparison outputs from the job array into per-species
tables + group-level aggregates.

Per-cell selection (which model wins each animal x stage) is already computed by
each array task, so this only concatenates and recomputes group rollups.

Usage:
    python scripts/scc/merge_cmp.py marmoset mouse human
    # reads  $OUTROOT/<species>/*/per_cell_comparison.csv   (OUTROOT=model_comparison_array)
    # writes model_comparison_<species>/{per_cell_comparison,win_counts_by_group,
    #         elpd_per_trial_by_group,selection_per_cell}.csv
"""
from __future__ import annotations

import os
import sys
import glob
import pandas as pd

OUTROOT = os.environ.get("OUTROOT", "model_comparison_array")
OUTDIR_FMT = os.environ.get("OUTDIR_FMT", "model_comparison_{species}")


def merge_species(species: str) -> None:
    files = sorted(glob.glob(os.path.join(OUTROOT, species, "*", "per_cell_comparison.csv")))
    if not files:
        print(f"[{species}] no per_cell files under {OUTROOT}/{species}/*/ — skipping")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    outdir = OUTDIR_FMT.format(species=species)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "per_cell_comparison.csv"), index=False)
    print(f"[{species}] merged {len(files)} animals -> {len(df)} rows")

    if "selected" in df.columns:
        sel_cols = [c for c in ["animal", "group", "stage", "winner_model",
                                "winner_elpd_diff", "winner_elpd_diff_se", "winner_decisive"]
                    if c in df.columns]
        df[df["selected"] == True][sel_cols].to_csv(
            os.path.join(outdir, "selection_per_cell.csv"), index=False)

        win = (df[df["selected"] == True].groupby(["group", "winner_model"]).size()
               .rename("n_cells_won").reset_index())
        win.to_csv(os.path.join(outdir, "win_counts_by_group.csv"), index=False)
        print(f"[{species}] === wins by group ===\n{win.to_string(index=False)}")

    if "elpd_per_trial" in df.columns:
        tbl = (df.groupby(["group", "model"])["elpd_per_trial"]
               .agg(["mean", "std", "count"]).reset_index())
        tbl.to_csv(os.path.join(outdir, "elpd_per_trial_by_group.csv"), index=False)
        print(f"[{species}] === mean ELPD/trial by group x model ===\n{tbl.to_string(index=False)}")


def main():
    for sp in (sys.argv[1:] or ["marmoset", "mouse", "human"]):
        merge_species(sp)


if __name__ == "__main__":
    main()
