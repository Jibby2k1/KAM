#!/usr/bin/env python3
"""Plot paired KAM–dot effects for MATH-500 and disjoint matched MATH items."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDIES = (("full", "MATH-500 · 500 items"), ("replication", "Balanced MATH study · 478 items"))
BACKBONES = (("smollm2", "SmolLM2"), ("llama3", "Llama 3"))
SLICES = (("full", "All items", "#253a50", "o"),
          ("integer", "Integer", "#7e4f26", "s"),
          ("noninteger", "Noninteger", "#166d6a", "D"))


def draw(directory, replication, output):
    datasets = {"full": directory, "replication": replication}
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": "#19242e", "axes.labelcolor": "#253a50",
        "axes.edgecolor": "#c8d3dc", "xtick.color": "#405468", "ytick.color": "#253a50",
    }):
        figure, axes = plt.subplots(1, 2, figsize=(10.3, 4.55), sharey=True, sharex=True)
        figure.subplots_adjust(left=.27, right=.975, top=.81, bottom=.26, wspace=.18)
        span = 0.0
        labels = [f"{name} · {group}" for _, name in BACKBONES for _, group, _, _ in SLICES]
        for ax, (study, title) in zip(axes, STUDIES, strict=True):
            ax.axvline(0, color="#728492", linewidth=1, linestyle="--")
            ax.axhline(2.5, color="#e1e8ed", linewidth=1)
            for i, (model, _) in enumerate(BACKBONES):
                summary = json.loads((datasets[study] / f"{model}-summary.json").read_text())
                for j, (label, _, color, mark) in enumerate(SLICES):
                    contrast = summary["groups"][label]["contrasts"]["full"]
                    center = 100 * contrast["mean_delta"]
                    low, high = [100 * bound for bound in contrast["paired_item_bootstrap95_given_seeds"]]
                    span = max(span, abs(low), abs(high))
                    ax.errorbar(center, i * 3 + j, xerr=[[center - low], [high - center]],
                                capsize=2.5, fmt=mark, color=color, markersize=5.5, linewidth=1.35)
            ax.set_title(title, fontweight="bold", pad=12)
            ax.grid(axis="x", color="#eef2f5")
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            for side in ("left", "right", "top"):
                ax.spines[side].set_visible(False)
        axes[0].set_yticks(range(6), labels)
        axes[0].set_ylim(5.65, -.55)
        axes[0].set_xlim(-span * 1.11, span * 1.11)
        figure.suptitle("KAM–dot differences by answer type", x=.08, y=.965, ha="left",
                        fontsize=17, fontweight="bold")
        figure.supxlabel("Difference in verified answer accuracy (percentage points)", x=.61, y=.118, fontsize=10.5)
        figure.text(.26, .048, "Intervals use paired items and condition on the three measured training seeds.",
                    color="#546476", fontsize=9)
        figure.savefig(output, dpi=170)
        plt.close(figure)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("full", type=Path)
    cli.add_argument("replication", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    draw(args.full, args.replication, args.output)
