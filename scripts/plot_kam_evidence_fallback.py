#!/usr/bin/env python3
"""Render layer-by-head MuSiQue attention coverage on the frozen holdout."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def draw(table, selection, output):
    kept = np.full((32, 32), np.nan)
    missed = np.zeros((32, 32))
    with table.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["panel"], row["query"], row["span"]) != ("holdout", "choice", "earlier_outside_first"):
                continue
            i, j = int(row["layer"])-1, int(row["head"])-1
            full, retained = float(row["dense_mass"]), float(row["retained_mass"])
            missed[i, j] = full-retained
            if full > 0:
                kept[i, j] = retained/full
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), layout="constrained")
    for ax, x, title, palette, label, vmax in (
        (axes[0], kept, "Retained fraction", "viridis", "Fraction of evidence mass", 1),
        (axes[1], missed, "Omitted mass", "magma", "Share of all dense attention", max(missed.max(), 1e-7)),
    ):
        image = ax.imshow(x, origin="upper", extent=(.5, 32.5, 32.5, .5), cmap=palette,
                          vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Query head")
        ax.set_xticks((1, 8, 16, 24, 32))
        ax.set_yticks((1, 8, 16, 24, 32))
        colorbar = fig.colorbar(image, ax=ax, shrink=.75, format=PercentFormatter(1))
        colorbar.set_label(label)
    axes[0].set_ylabel("Llama layer")
    layers = [i+1 for i in selection["flash_layers_zero_based"]]
    axes[0].scatter(np.zeros(len(layers))+.24, layers, marker="s", s=8, clip_on=False,
                    color="#175a9a", label="Flash layer selected on dev")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False)
    fig.suptitle("Earlier MuSiQue evidence beyond the first block", x=.05, ha="left", weight="bold")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("layer_head_csv", "selection", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    draw(a.layer_head_csv, json.loads(a.selection.read_text()), a.output)
