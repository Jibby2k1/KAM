#!/usr/bin/env python3
"""Visualize the sequential-only successes and the costs of switching."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text())
    test = result["test"]["all"]["policies"]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kam-sequential-fallback",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "text.color": "#182238", "axes.labelcolor": "#182238"})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.1), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1]})
    names = ("Complete support", "Joint EM")
    for y, metric in enumerate(("support", "joint_exact")):
        potential = test["sequential"]["vs_broad"][metric]["wins"]
        vetoed = result["diagnostics"][metric + "_rescues_vetoed"]
        visible = potential - vetoed
        threshold = test["five_percent_diagnostic"]["vs_broad"][metric]
        wins, losses = threshold["wins"], threshold["losses"]
        for left, width, color, legend in ((0, vetoed, "#748199", "Vetoed"),
                                            (vetoed, visible - wins, "#4e86be", "Below threshold"),
                                            (potential - wins, wins, "#127069", "Recovered")):
            axes[0].barh(y, width, left=left, color=color, height=.45, label=legend if y == 0 else None)
            if width:
                axes[0].text(left + width / 2, y, str(width), color="white", ha="center", va="center",
                             fontsize=9, weight="bold")
        axes[1].barh(y, wins, color="#127069", height=.45, label="Wins" if y == 0 else None)
        axes[1].barh(y, -losses, color="#bc5142", height=.45, label="Losses" if y == 0 else None)
        if wins:
            axes[1].text(wins + .13, y, f"+{wins}", ha="left", va="center")
        if losses:
            axes[1].text(-losses - .13, y, f"−{losses}", ha="right", va="center")
    headroom = max(test["sequential"]["vs_broad"][metric]["wins"] for metric in ("support", "joint_exact"))
    extent = max(max(d["wins"], d["losses"]) for d in test["five_percent_diagnostic"]["vs_broad"].values()) + 1
    axes[0].set(title="Where the sequential rescues went", xlabel="Questions", xlim=(0, headroom + 2))
    axes[1].set(title="Effect of the diagnostic switches", xlabel="Questions relative to broad", xlim=(-extent, extent))
    axes[1].axvline(0, color="#748199", linewidth=.75, zorder=0)
    for ax in axes:
        ax.set_yticks([0, 1], names)
        ax.set_ylim(1.55, -.75)
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=len(ax.get_legend_handles_labels()[0]),
                  fontsize=9)
    fig.savefig(args.output, metadata={"Date": None, "Title": "Sequential fallback on fresh HotpotQA pools"})
    if args.output.suffix == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
