#!/usr/bin/env python3
"""Plot support and joint answer outcomes for the two fixed choices and the gate."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())["tests"]
    cohorts = ("all", "natural_linked_distractor", "gold_unlinked")
    labels = ("All questions", "Natural linked\ndistractors", "Unlinked\nsupports")
    arms = (("broad", "Broad reranker", "#52769e"), ("linked", "Linked-only reranker", "#a65337"),
            ("learned_gate", "Learned gate", "#007f72"), ("oracle_choice", "Two-reranker oracle", "#9ba8b8"))
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "svg.fonttype": "none", "svg.hashsalt": "kam-native-routing-gate"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, layout="constrained")
    x = np.arange(len(cohorts))
    for ax, metric, title in zip(axes, ("support", "joint_exact"), ("Both supports found", "Joint exact match"), strict=True):
        for i, (arm, name, color) in enumerate(arms):
            rates = [data[c]["correct"][arm][metric] / data[c]["n"] for c in cohorts]
            ax.bar(x + (i - 1.5) * .19, rates, width=.17, label=name, color=color)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, [f"{label}\n(n={data[key]['n']:,})" for key, label in zip(cohorts, labels, strict=True)])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce5ee")
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", ncols=4, frameon=False, fontsize=9)
    fig.suptitle("Routing on fixed Wikipedia pools", x=.02, ha="left", fontweight="bold", fontsize=16)
    fig.savefig(args.output, bbox_inches="tight", metadata={"Date": None})
    args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
