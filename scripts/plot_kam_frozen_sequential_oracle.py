#!/usr/bin/env python3
"""Visualize how much correct evidence and answer recovery the two methods share."""

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
    data = json.loads(a.summary.read_text())["tests"]
    labels = (("all", "All questions"), ("natural_linked_distractor", "Natural linked distractors"),
              ("gold_unlinked", "Unlinked supports"))
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                                "svg.hashsalt": "kam-frozen-sequential-oracle"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.65), sharey=True, layout="constrained")
    y = np.arange(len(labels))
    for ax, metric, title in zip(axes, ("support", "joint_exact"), ("Both supports recovered", "Joint exact match"), strict=True):
        cases = [data[key] for key, _ in labels]
        broad = np.array([c["exclusive_successes"][metric]["broad_only"] / c["n"] for c in cases])
        seq = np.array([c["exclusive_successes"][metric]["sequential_only"] / c["n"] for c in cases])
        shared = np.array([c["correct"]["broad"][metric] / c["n"] for c in cases]) - broad
        ax.barh(y, shared, color="#9ba8b8", height=.46, label="Both methods")
        ax.barh(y, broad, left=shared, color="#52769e", height=.46, label="Broad only")
        ax.barh(y, seq, left=shared+broad, color="#007f72", height=.46, label="Sequential only")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce5ee")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_yticks(y, [f"{name}\n(n={data[key]['n']:,})" for key, name in labels])
    axes[0].invert_yaxis()
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, ncols=3, loc="outside lower center", frameon=False)
    fig.suptitle("Where sequential retrieval adds recoverable cases", x=.02, ha="left",
                 fontweight="bold", fontsize=16)
    fig.savefig(a.output, bbox_inches="tight", metadata={"Date": None})
    a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
