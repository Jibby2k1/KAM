#!/usr/bin/env python3
"""Plot selective precision on daily Eurostat cutoffs with frozen priors."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#275d88"
TEAL = "#007a73"
INK = "#152632"
MUTE = "#526777"
BG = "#f8faf9"


def risk_curve(rows, case, policy):
    values = [row["outcomes"][case]["policies"][policy] for row in rows
              if row["stage"] == "original_test" and row["outcomes"][case]["alternatives"] > 1]
    grouped = defaultdict(Counter)
    for row in values:
        grouped[row["confidence"]].update(n=1, correct=row["strict_exact"])
    n, correct = 0, 0
    x, y = [], []
    for threshold in sorted(grouped, reverse=True):
        data = grouped[threshold]
        n += data["n"]
        correct += data["correct"]
        x.append(100*n/len(values))
        y.append(100*correct/n)
    return x, y


def draw(root, output):
    summary = json.loads((root/"summary.json").read_bytes())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.55), facecolor=BG, sharey=True)
    chart = "fortnight/bucket_only"
    for ax, name, label in zip(axes, ("unemployment", "industry"), ("Unemployment", "Industrial production"), strict=True):
        source = json.loads(gzip.decompress((root/f"{name}-daily.json.gz").read_bytes()))
        results = summary[name]["daily"]["matched_calibration"][chart]["original_test"]
        total = results["n"] - results["strict_unique"]
        ax.set_title(f"{label}\n{total:,} ambiguous lookups", loc="left", color=INK, fontsize=13)
        ax.set_facecolor(BG)
        ax.set_axisbelow(True)
        ax.grid(color="#dbe5e8")
        for frame in ax.spines.values():
            frame.set_visible(False)
        for policy, color, title in (("uniform", BLUE, "Uniform days"), ("weekday_monthday", TEAL, "Calendar prior")):
            ax.plot(*risk_curve(source, chart, policy), color=color, linewidth=1.8, label=title)
            scores = results["policies"][policy]
            n = scores["accepted_ambiguous"]
            if n:
                ax.scatter(100*n/total, 100*scores["accepted_ambiguous_correct"]/n,
                           color=color, edgecolor="white", s=65, zorder=4)
        ax.axhline(95, color=MUTE, linestyle="--", linewidth=.8)
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Share of ambiguous lookups accepted (%)", color=MUTE)
        ax.tick_params(colors=MUTE, length=0)
    axes[0].set_ylabel("Strict snapshots correct among accepted (%)", color=MUTE)
    axes[1].legend(frameon=False, labelcolor=MUTE, loc="lower left")
    fig.subplots_adjust(left=.085, right=.965, top=.71, bottom=.22, wspace=.20)
    fig.text(.085, .935, "A release calendar can improve selective lookup", fontsize=19, weight="bold", color=INK)
    fig.text(.085, .86, "Fortnight publication buckets · daily-cutoff evaluation · original 2024–25 panel", fontsize=10, color=MUTE)
    fig.text(.085, .062, "Dots use thresholds set on 2023 daily probes. Uniform unemployment declined every ambiguous lookup at that setting.",
             fontsize=8, color=MUTE)
    fig.savefig(output, dpi=165, facecolor=BG)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    draw(a.results, a.output)


if __name__ == "__main__":
    main()
