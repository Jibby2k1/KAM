#!/usr/bin/env python3
"""Plot low-rate benchmark retention against measured validation KL."""
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BACKBONES = (("smollm2", "SmolLM2 1.7B Instruct"), ("llama3", "Llama 3 8B Instruct"))
ARMS = {"kam": "#166768", "dot": "#a65025"}
MARKERS = {"lower_lr": "o", "lower_lr_kl05": "^", "lower_lr_kl": "s"}


def chart(directory, output):
    with plt.rc_context({
        "svg.fonttype": "none", "svg.hashsalt": "kam-low-rate",
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#19242e",
        "axes.labelcolor": "#19242e", "xtick.color": "#324352", "ytick.color": "#324352",
    }):
        fig, plots = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=.095, right=.98, top=.80, bottom=.27, wspace=.17)
        for (model, label), ax in zip(BACKBONES, plots, strict=True):
            official = json.loads((directory / f"{model}-summary.json").read_text())
            diag = directory / "diagnostics"
            matched = json.loads((diag / f"{model}-low-rate-matched-kl-summary.json").read_text())
            pilot_file = diag / f"{model}-low-rate-kl05-summary.json"
            pilot = json.loads(pilot_file.read_text()) if pilot_file.exists() else None
            untouched = official["tasks"]["gsm8k"]["untouched"]["score"]
            for arm, color in ARMS.items():
                for recipe, marker in MARKERS.items():
                    summary = pilot if recipe == "lower_lr_kl05" else official
                    if summary is None:
                        continue
                    nll = [fit["training"][-1]["forward_kl"] for fit in summary["training"].values()
                           if fit["recipe"] == recipe and fit["arm"] == arm]
                    assert len(nll) == 3 and min(nll) > 0
                    score = summary["tasks"]["gsm8k"]["seed_means"][f"{recipe}-{arm}"]["mean"]
                    ax.scatter(mean(nll), (score - untouched) * 100, color=color, marker=marker, s=49, zorder=3)

            cal = matched["seed_details"].values()
            cap = mean(r["validation_dot_matched"] for r in cal)
            delta = matched["contrast"]["matched-minus-untouched"]["mean_delta"] * 100
            ax.scatter(cap, delta, color=ARMS["dot"], marker="*", s=123, zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(.002, 1.4)
            ax.set_xticks((.003, .01, .1, 1), ("0.003", "0.01", "0.1", "1"))
            ax.axhline(0, color="#5c6e7d", linewidth=.9)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e9edf1", linewidth=.8)
            ax.tick_params(axis="both", length=0)
            ax.set_title(label, fontweight="bold", pad=10)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle("Validation drift and GSM8K at the lower learning rate", x=.095, y=.967,
                     ha="left", fontsize=17, fontweight="bold")
        fig.supylabel("Change from untouched (percentage points)", x=.024, y=.53, fontsize=10.5)
        fig.supxlabel("Raw-text validation KL (nats per token, log scale)", x=.543, y=.177, fontsize=10.5)
        labels = ["KAM", "Dot", "KL weight 0", "KL weight 0.5", "KL weight 4", "Dot, scaled after fit"]
        handles = [Line2D([0], [0], color=c, linewidth=2) for c in ARMS.values()]
        handles += [Line2D([0], [0], color="#52616d", marker=m, linestyle="None") for m in MARKERS.values()]
        handles.append(Line2D([0], [0], color=ARMS["dot"], marker="*", linestyle="None", markersize=9))
        fig.legend(handles, labels, ncols=3, frameon=False, loc="lower left",
                   bbox_to_anchor=(.08, .005), handlelength=1.4, columnspacing=1.5)
        fig.savefig(output, dpi=180, metadata={"Date": None} if output.suffix == ".svg" else None)
        plt.close(fig)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    chart(args.results, args.output)
