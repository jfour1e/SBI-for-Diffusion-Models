#!/usr/bin/env python
"""Render a corner / pairplot of the inferred posterior for one animal+stage.

Loads `group_outputs_ar/animals/<ANIMAL>/<STAGE>/posterior_combined.npy`
(or whatever OUTDIR you used) and writes a `posterior_pairplot.png` next to it.

Env vars:
  ANIMAL       required (e.g. Churro)
  STAGE        default "70-30"
  MODEL_NAME   default "lapse_noleak_ar" (used only for axis labels)
  OUTDIR       default "group_outputs_ar"
  ANIMALS      comma-separated list of animals (overrides ANIMAL)
  STAGES       comma-separated list of stages (overrides STAGE)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

from sbi.analysis import pairplot

from sbi_for_diffusion_models.model_specs import select_model


def plot_one(animal: str, stage: str, outdir: str, param_names: list[str]) -> str:
    cell_dir = os.path.join(outdir, "animals", animal, stage)
    samples_path = os.path.join(cell_dir, "posterior_combined.npy")
    if not os.path.exists(samples_path):
        print(f"  [skip] {animal}/{stage}: no posterior_combined.npy at {samples_path}")
        return ""

    samples = np.load(samples_path)
    if samples.shape[1] != len(param_names):
        print(
            f"  [skip] {animal}/{stage}: posterior has {samples.shape[1]} cols, "
            f"but MODEL_NAME implies {len(param_names)} params {param_names}"
        )
        return ""

    fig, _ = pairplot(
        torch.from_numpy(samples),
        labels=param_names,
        figsize=(2.0 * len(param_names), 2.0 * len(param_names)),
    )
    fig.suptitle(f"Posterior — {animal}  ({stage})  N={samples.shape[0]}", y=1.01, fontsize=12)

    out_path = os.path.join(cell_dir, "posterior_pairplot.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {animal:14s} {stage:10s}  ->  {out_path}")
    return out_path


def main():
    MODEL_NAME = os.environ.get("MODEL_NAME", "lapse_noleak_ar")
    OUTDIR = os.environ.get("OUTDIR", "group_outputs_ar")

    if "ANIMALS" in os.environ:
        animals = [a.strip() for a in os.environ["ANIMALS"].split(",") if a.strip()]
    else:
        if "ANIMAL" not in os.environ:
            print("ERROR: set ANIMAL=<name> or ANIMALS=<a,b,c>", file=sys.stderr)
            sys.exit(1)
        animals = [os.environ["ANIMAL"]]

    if "STAGES" in os.environ:
        stages = [s.strip() for s in os.environ["STAGES"].split(",") if s.strip()]
    else:
        stages = [os.environ.get("STAGE", "70-30")]

    _, _, param_names, _, _ = select_model(MODEL_NAME)
    param_names = list(param_names)

    for animal in animals:
        for stage in stages:
            plot_one(animal, stage, OUTDIR, param_names)


if __name__ == "__main__":
    main()
