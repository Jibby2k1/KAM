#!/usr/bin/env python3
"""Plot held-out complete evidence recovery across matched text perturbations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    data = json.loads(a.summary.read_text())["tests"]
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                                "svg.fonttype": "none", "svg.hashsalt": "kam-coverage-gate"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained", sharey=True)
    variants = ("clean", "sham", "linked")
    colors = (("learned", "Learned gate", "#007f72", "o"),
              ("full", "Always use coverage", "#2268a0", "s"),
              ("without", "Omit coverage", "#9a5340", "D"))
    for ax, (key, title) in zip(axes, (("twowiki-test", "2Wiki · 420 questions"),
                                      ("musique-test", "MuSiQue · 300 questions")), strict=True):
        cohort = data[key]
        for policy, label, color, marker in colors:
            fractions = [cohort[v]["correct"][policy]/cohort[v]["n"] for v in variants]
            ax.plot(range(3), fractions, label=label, color=color, marker=marker, linewidth=2, markersize=5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(3), ["Clean", "Sham text", "False link"])
        ax.set_ylim(0,.84)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y",color="#e1e7ef")
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].set_ylabel("Both supporting documents recovered")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("A learned gate helps clean retrieval on 2Wiki", x=.02, ha="left", fontweight="bold", fontsize=16)
    fig.savefig(a.output, bbox_inches="tight", metadata={"Date": None})
    a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
