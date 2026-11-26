#!/usr/bin/env python3
"""Plot matched archive masks and sentence-verifier transfer."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def finish(figure, output, title):
    figure.suptitle(title, x=.02, ha="left", fontweight="bold", fontsize=16)
    figure.savefig(output, bbox_inches="tight", metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(figure)


def style(ax, title):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce5ee")
    ax.spines[["top", "right", "left"]].set_visible(False)


def archive_plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.2), sharey=True, layout="constrained")
    specs = (("observed", "48 visible facts", "#52769e"),
             ("three_releases", "Mask same-address releases", "#0f8074"),
             ("load_control", "Mask equally many distractors", "#a65337"))
    clocks = np.arange(2)
    for ax, prefix, title in zip(axes, ("dot", "radial"), ("Dot attention", "Radial attention"), strict=True):
        for i, (panel, label, color) in enumerate(specs):
            rows = [summary["performance"][f"{prefix}_without_absolute:{c}:{panel}"] for c in (1, 3)]
            accuracy = [r["joint"] / r["n"] for r in rows]
            ax.bar(clocks + (i - 1) * .235, accuracy, width=.215, label=label, color=color)
        ax.set_xticks(clocks, ["Observed offsets", "Publication offsets ×3"])
        style(ax, title)
    axes[0].set_ylabel("Exact address and revision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="outside lower center", frameon=False, fontsize=9)
    finish(fig, output, "Dallas Fed revision retrieval under matched masks")


def verifier_plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.4), sharey=True, layout="constrained")
    variants = ("clean", "sham", "linked", "contextual_sham", "contextual_linked")
    labels = ("Clean", "Sham", "False link", "Contextual\nsham", "Contextual\nfalse link")
    arms = (("full", "Frozen context head", "#52769e", "s"),
            ("base_reranker", "Base reranker", "#a65337", "D"),
            ("semantic_sentence", "Sentence reranker", "#0f8074", "o"))
    for ax, split, title in zip(axes, ("twowiki-test", "musique-test"), ("2Wiki", "MuSiQue"), strict=True):
        rows = [summary["tests"][f"{split}:{v}"] for v in variants]
        for arm, label, color, marker in arms:
            accuracy = [r["correct"][arm] / r["n"] for r in rows]
            ax.plot([0, 1, 2], accuracy[:3], marker=marker, color=color, label=label, markersize=5)
            ax.plot([3, 4], accuracy[3:], marker=marker, color=color, markersize=5)
        ax.set_xticks(range(5), labels)
        style(ax, title)
    axes[0].set_ylabel("Both supporting documents recovered")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    finish(fig, output, "Sentence evidence under clean and false-link inputs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-summary", type=Path)
    p.add_argument("--verifier-summary", type=Path)
    p.add_argument("--figures", type=Path, required=True)
    a = p.parse_args()
    if not (a.archive_summary or a.verifier_summary):
        p.error("Provide an archive or verifier summary")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                                "svg.hashsalt": "kam-revision-and-sentence-followups"})
    if a.archive_summary:
        archive_plot(json.loads(a.archive_summary.read_text()), a.figures / "oecd_masks.svg")
    if a.verifier_summary:
        verifier_plot(json.loads(a.verifier_summary.read_text()), a.figures / "sentence_verifier.svg")


if __name__ == "__main__":
    main()
