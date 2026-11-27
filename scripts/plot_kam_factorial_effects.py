#!/usr/bin/env python3
"""Plot the paired high-rate KL effect from the verified factorial summaries."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BENCHMARKS = {
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "gpqa": "GPQA Diamond",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
}
STYLES = {"kam": ("#166768", "o", -0.13), "dot": ("#a65025", "D", 0.13)}


def plot(paths, output):
    models = [json.loads(path.read_text()) for path in paths]
    names = ["SmolLM2 1.7B Instruct", "Llama 3 8B Instruct"]
    with plt.rc_context({
        "svg.fonttype": "none", "svg.hashsalt": "kam-factorial",
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#19242e",
        "axes.labelcolor": "#19242e", "xtick.color": "#324352", "ytick.color": "#324352",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.155, right=.975, bottom=.23, top=.795, wspace=.15)
        for name, model, ax in zip(names, models, axes, strict=True):
            ax.set_title(name, pad=11, fontweight="bold")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e9edf1", linewidth=.8)
            ax.axvline(0, color="#5c6e7d", linewidth=.9)
            ax.set_yticks(range(len(BENCHMARKS)), BENCHMARKS.values())
            ax.set_ylim(len(BENCHMARKS) - .55, -.45)
            ax.tick_params(axis="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            for index, task in enumerate(BENCHMARKS):
                for arm, (color, marker, shift) in STYLES.items():
                    contrast = model["tasks"][task]["contrasts"][f"{arm}:kl-at-high-lr"]
                    mean = contrast["mean_delta"] * 100
                    low, high = (v * 100 for v in contrast["paired_item_bootstrap95_given_seeds"])
                    ax.errorbar(mean, index + shift, xerr=[[mean-low], [high-mean]],
                                fmt=marker, color=color, capsize=3, markersize=5.5, linewidth=1.5)

        fig.suptitle("Effect of adding KL at the higher learning rate", x=.155, y=.965,
                     ha="left", fontsize=17, fontweight="bold")
        fig.supxlabel("Score change (percentage points)", x=.565, y=.125, fontsize=11)
        handles = [Line2D([0], [0], color=color, marker=marker, linewidth=1.5)
                   for color, marker, _ in STYLES.values()]
        fig.legend(handles, ["KAM", "Dot"], loc="lower left", bbox_to_anchor=(.14, .012),
                   ncols=2, frameon=False, handlelength=1.7, columnspacing=1.3)
        fig.text(.48, .046, "95% paired item intervals, conditional on three fixed seeds", fontsize=9,
                 color="#435464")
        fig.savefig(output, dpi=180, metadata={"Date": None} if output.suffix == ".svg" else None)
        plt.close(fig)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smollm2", type=Path)
    parser.add_argument("llama3", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot((args.smollm2, args.llama3), args.output)
