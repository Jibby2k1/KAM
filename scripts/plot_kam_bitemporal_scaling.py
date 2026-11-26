#!/usr/bin/env python3
"""Render the paired KAM decoder length-generalization plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, NullLocator


def read(path):
    return json.loads(path.read_text())


def show(ax, depths):
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths, [str(n) for n in depths])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.025, 1.025)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", color="#dce2ec", linewidth=0.8)
    ax.set_xlabel("Records in each candidate bank")
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#a4afbf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--curriculum", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    conf, followup = read(a.confirmation), read(a.curriculum)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "svg.fonttype": "none", "svg.hashsalt": "kam-bitemporal-scaling",
                                "axes.labelcolor": "#172237",
                                "xtick.color": "#344359", "ytick.color": "#344359"})
    fig, (A, B) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    depths = (20, 32, 64, 128)
    methods = (
        ("dot_context", "Dot, absolute position", "#55708e", "--", "o"),
        ("radial_context", "Radial, absolute position", "#9a613f", "--", "^"),
        ("dot_without_absolute", "Dot, no absolute position", "#26598c", "-", "o"),
        ("radial_without_absolute", "Radial, no absolute position", "#008379", "-", "^"),
    )
    for arm, name, color, line, marker in methods:
        rows = [conf["slices"][f"mixed:{n}"][arm] for n in depths]
        A.plot(depths, [r["joint"] for r in rows], label=name, color=color, linestyle=line,
               marker=marker, linewidth=2, markersize=4)
        if arm.endswith("without_absolute"):
            A.fill_between(depths, [min(r["seed_joint"].values()) for r in rows],
                           [max(r["seed_joint"].values()) for r in rows], color=color, alpha=.12)
    A.set_title("Position controls · six paired seeds", loc="left", fontweight="bold")
    A.set_ylabel("Exact alias and payload accuracy")
    A.legend(loc="lower right", bbox_to_anchor=(.99, .10), frameon=False, fontsize=8.4)
    show(A, depths)

    depths = (20, 64, 128, 256, 512)
    choices = (
        ("radial_without_absolute/no-clock/short", "Train through 20, relative clocks", "#55708e", "--"),
        ("radial_without_absolute/wide", "Train through 64, absolute query clocks", "#b45432", "-"),
        ("radial_without_absolute/no-clock/wide", "Train through 64, relative clocks", "#008379", "-"),
    )
    for arm, name, color, line in choices:
        rows = [followup["models"][arm]["slices"][f"mixed:{n}"] for n in depths]
        B.plot(depths, [r["joint"] for r in rows], label=name, color=color, linestyle=line,
               marker="o", linewidth=2, markersize=4)
    B.set_title("Equal-token training · four fresh seeds", loc="left", fontweight="bold")
    B.legend(loc="lower left", frameon=False, fontsize=8.4)
    show(B, depths)
    fig.suptitle("Bitemporal retrieval under longer, noisy histories", x=.023, ha="left",
                 fontweight="bold", color="#172237", fontsize=15)
    svg = a.output.suffix.lower() == ".svg"
    fig.savefig(a.output, dpi=170, bbox_inches="tight", metadata={"Date": None} if svg else None)
    if svg:
        a.output.write_text("\n".join(line.rstrip() for line in a.output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
