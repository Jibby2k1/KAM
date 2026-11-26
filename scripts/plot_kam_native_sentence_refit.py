#!/usr/bin/env python3
"""Plot held-out support recovery relative to the frozen context scorer."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    tests = json.loads(a.summary.read_text())["tests"]
    models = (
        ("transferred_2wiki", "2Wiki paired fit", "#a65337"),
        ("transferred_2wiki_clean", "2Wiki clean-only fit", "#a65337"),
        ("native_base", "Native pair calibration", "#52769e"),
        ("native_semantic", "Native sentence reranker", "#0f8074"),
        ("native_linked_semantic", "Native linked-only fit", "#0f8074"),
    )
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                                "font.size": 10, "svg.hashsalt": "kam-native-refit"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    cohorts = (("natural_linked_distractor", "Natural linked distractors"),
               ("gold_unlinked", "Gold support pair unlinked"))
    for ax, (key, title) in zip(axes, cohorts, strict=True):
        panel = tests[key]
        for idx, (arm, _, color) in enumerate(models):
            d = panel["vs_frozen_context"][arm]
            point = 100 * d["delta"]
            low, high = (100 * x for x in d["bootstrap95"])
            ax.errorbar(point, idx, xerr=[[point-low], [high-point]], fmt="o", capsize=3, color=color)
        ax.axvline(0, color="#8893a0", linewidth=1)
        ax.set_title(f"{title} · {panel['n']} questions", loc="left", fontweight="bold")
        ax.set_xlabel("Complete support recovery (percentage-point change)")
        ax.grid(axis="x", color="#e0e6ee")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(range(len(models)), [r[1] for r in models])
    axes[0].invert_yaxis()
    fig.suptitle("Reranker transfer and native retraining", x=.02, ha="left", fontweight="bold", fontsize=15)
    fig.savefig(a.output, bbox_inches="tight", metadata={"Date": None})
    a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
