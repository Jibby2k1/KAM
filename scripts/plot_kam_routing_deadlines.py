#!/usr/bin/env python3
"""Plot the matched quality-deadline curves from the short-mirror comparison."""
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary, destination):
    data = json.loads(summary.read_text())
    questions = data["contrasts"]["train"]["questions"]
    routes = (("lexical", "BM25 + one view", "#195d65", "o", "-"),
              ("lexical_mirrored", "BM25 + mirror", "#a24d20", "s", "-"),
              ("mirrored", "Mirror both earlier passages", "#62708b", "^", "--"))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#19242e"}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
        fig.subplots_adjust(left=.115, right=.98, top=.77, bottom=.18, wspace=.15)
        for ax, arm, name in zip(axes, ("kam", "dot"), ("KAM", "Dot attention"), strict=True):
            for route, label, color, marker, style in routes:
                rows = [result["train"][route]["all_four_correct_at_ms"]
                        for key, result in data["checkpoints"].items() if key.endswith("-" + arm)]
                assert len(rows) == 3
                deadlines = sorted(map(int, rows[0]))
                rates = [100 * mean(row[str(deadline)] for row in rows) / questions for deadline in deadlines]
                ax.plot(deadlines, rates, color=color, marker=marker, linestyle=style, linewidth=2.2, label=label)
            ax.set_title(name, loc="left", fontweight="bold")
            ax.set_xticks((15, 40, 60, 100, 150))
            ax.set_yticks(range(0, 101, 25))
            ax.set_ylim(-2, 102)
            ax.grid(axis="y", color="#e6eaf0", linewidth=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#bbc8d2")
            ax.tick_params(length=0)
        axes[0].set_ylabel("Questions solved in all four placements (%)")
        fig.suptitle("The deadline changes which route works best", fontweight="bold", fontsize=17,
                     x=.06, y=.975, ha="left")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.053, .91),
                   frameon=False, ncols=3, columnspacing=1.3)
        fig.supxlabel("Deadline per presentation (ms)", y=.068, fontsize=10.5)
        fig.savefig(destination, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.destination)
