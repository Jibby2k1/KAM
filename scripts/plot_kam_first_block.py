#!/usr/bin/env python3
"""Plot retained dense attention mass and first-block omissions by Llama layer."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def draw(summary, output):
    layers = range(1, 33)
    table = summary["mass"]["combined"]["by_layer"]
    rows = [table[str(i)]["pruned"] for i in layers]
    fig, (mass, anchor) = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, layout="constrained")
    color = ("#A54820", "#155B87")
    for ax, series in ((mass, (("ordinary", "Ordinary eight-block shortlist"), ("protected", "First block protected"))),
                       (anchor, (("first_missing_rate", "First block omitted"),
                                 ("first_mass_when_missing", "First-block mass when omitted")))):
        for hue, (metric, label) in zip(color, series, strict=True):
            ax.plot(layers, [r[metric] for r in rows], color=hue, label=label, linewidth=2, marker="o", markersize=3)
        ax.set_xlim(1, 32)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#DDE3E8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="lower right", frameon=True, fontsize=9)
    mass.set_title("How much dense attention mass does the shortlist keep?", loc="left", weight="bold")
    mass.set_ylabel("Dense mass retained")
    anchor.set_ylabel("Queries or attention mass")
    anchor.set_xlabel("Llama layer (one-based)")
    anchor.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    draw(json.loads(args.summary.read_text()), args.output)
