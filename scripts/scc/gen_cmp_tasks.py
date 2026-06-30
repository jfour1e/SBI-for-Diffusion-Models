#!/usr/bin/env python
"""Generate a TSV task list for the model-comparison job array.

Each line is `<species>\t<animal>`; one array task per line handles all of that
animal/subject's cells (stages x models). Roster definitions mirror the
per-species adapters in scripts/model_comparison.py.

Usage:
    python scripts/scc/gen_cmp_tasks.py marmoset mouse > scripts/scc/cmp_tasks_mm.tsv
    python scripts/scc/gen_cmp_tasks.py human          > scripts/scc/cmp_tasks_human.tsv
"""
from __future__ import annotations

import sys
import pandas as pd

ALL = "/projectnb/depaqlab/rsenne/sbi-python/SBI-for-Diffusion-Models/ALLspecies_trials_combined.csv"
MARM = "/projectnb/ssmsvi/rsenne/data_marmoset/marmoset_data.csv.gz"

ROSTERS = {
    "marmoset": dict(path=MARM, specie=None, stage_col="stage",
                     stages=["100-0", "90-10", "80-20", "70-30", "60-40"]),
    "mouse":    dict(path=ALL, specie="Mouse", stage_col="stage_cat",
                     stages=["100-0", "90-10", "80-20"]),
    "human":    dict(path=ALL, specie="Human", stage_col="stage_cat",
                     stages=["test"]),
}


def animals_for(species: str) -> list[str]:
    r = ROSTERS[species]
    df = pd.read_csv(r["path"], compression="infer",
                     dtype={"flashes_left": str, "flashes_right": str}, low_memory=False)
    if r["specie"] is not None:
        df = df[df["specie"] == r["specie"]]
    df = df[df[r["stage_col"]].isin(r["stages"])]
    return sorted(df["name"].dropna().unique().tolist())


def main():
    species_list = sys.argv[1:] or ["marmoset", "mouse"]
    for sp in species_list:
        if sp not in ROSTERS:
            raise SystemExit(f"unknown species {sp!r}; options {sorted(ROSTERS)}")
        for animal in animals_for(sp):
            print(f"{sp}\t{animal}")


if __name__ == "__main__":
    main()
