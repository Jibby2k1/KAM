#!/usr/bin/env python3
"""Plot independent-archive exact joint accuracy under paired interventions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    comparison = json.loads(a.summary.read_text())["tests"]["48"]["performance"]
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.fonttype": "none", "svg.hashsalt": "kam-oecd-transfer"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained", sharey=True)
    panels = ("archive", "archive_scaled", "archive_thin", "archive_thin_scaled")
    x = np.arange(len(panels))
    lines = (("procedural_dual_boundary", "Boundary coverage", "#839bb3"),
             ("procedural_archive_topology", "Synthetic release clusters", "#007f72"),
             ("archive", "Philadelphia archive fit", "#b86a4a"))
    for ax, (geometry, title) in zip(axes, (("dot", "Dot attention"), ("radial", "Radial attention")), strict=True):
        for i, (source, label, color) in enumerate(lines):
            rows = [comparison[f"{geometry}_without_absolute:{source}:{panel}"]["all"] for panel in panels]
            ax.bar(x + (i-1)*.24, [r["joint"]/r["n"] for r in rows], width=.21, label=label, color=color)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, ["Observed", "Observed\n3× offsets", "Three\nreleases", "Three releases\n3× offsets"])
        ax.set_ylim(0,1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y",color="#e0e6ee")
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].set_ylabel("Exact address and revision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle("Transfer to the Dallas Fed’s OECD revision archive", x=.02, ha="left",
                 fontweight="bold", fontsize=16)
    fig.savefig(a.output, bbox_inches="tight", metadata={"Date": None})
    a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
