#!/usr/bin/env python3
"""Show where full-corpus second-hop search recovers a missing support."""

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
    cohort = result["cohorts"]["one_absent_first_hop_gold"]
    labels = ("Missing support in top 64", "Both supports selected", "Joint exact match")
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "kam-conditioned-corpus",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "text.color": "#182238"})
    fig, ax = plt.subplots(figsize=(7.8, 3.1), layout="constrained")
    maximum = 0
    for arm, name, color, offset in (
        ("corpus_unconditioned", "Unconditioned corpus search", "#748199", -.16),
        ("corpus_conditioned", "Conditioned corpus search", "#127069", .16),
    ):
        counts = (cohort["corpus_expansion"][arm]["first_plus_top64_both_gold"],
                  cohort["correct"][arm]["support"], cohort["correct"][arm]["joint_exact"])
        maximum = max(maximum, *counts)
        for i, count in enumerate(counts):
            y = i + offset
            ax.barh(y, count, height=.27, color=color, label=name if i == 0 else None)
            ax.text(count + max(cohort["n"] / 100, 1), y, str(count), va="center", color=color, weight="bold")
    ax.set(title=f"Second-hop recovery with a correct first hop ({cohort['n']} questions)",
           xlabel="Questions", xlim=(0, maximum * 1.13 + 4), ylim=(2.55, -.55))
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2)
    fig.savefig(args.output, metadata={"Date": None, "Title": "Conditioned HotpotQA corpus search"})
    if args.output.suffix == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
