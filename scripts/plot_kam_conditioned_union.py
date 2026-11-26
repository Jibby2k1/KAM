#!/usr/bin/env python3
"""Show where recovered corpus evidence survives reranking on incomplete pools."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    cohort = json.loads(a.summary.read_text())["cohorts"]["incomplete_original"]
    labels = ("Both supports in union", "Gold pair shortlisted", "Both supports selected", "Joint exact match")
    metrics = ("covered", "gold_pair_in_shortlist", "support", "joint_exact")
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kam-conditioned-union", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "text.color": "#182238"})
    fig, ax = plt.subplots(figsize=(8.3, 3.6), layout="constrained")
    peak = max(cohort["stages"][arm]["covered"] for arm in ("unconditioned_union", "conditioned_union"))
    for arm, label, color, shift in (("unconditioned_union", "Unconditioned expansion", "#65738d", -.16),
                                      ("conditioned_union", "Conditioned expansion", "#127069", .16)):
        for i, metric in enumerate(metrics):
            count = cohort["stages"][arm][metric]
            y = i + shift
            ax.barh(y, count, color=color, height=.27, label=label if i == 0 else None)
            ax.text(count + max(1, peak / 80), y, str(count), va="center", color=color, weight="bold")
    ax.set(title=f"Evidence recovery from incomplete pools ({cohort['n']} questions)", xlabel="Questions",
           xlim=(0, peak * 1.15 + 4), ylim=(3.55, -.55))
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
    fig.savefig(a.output, metadata={"Date": None, "Title": "Evidence recovery after corpus expansion"})
    if a.output.suffix == ".svg":
        a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
