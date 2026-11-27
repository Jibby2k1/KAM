#!/usr/bin/env python3
"""Plot matched request-time effects of index validation and KV allocation."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LENGTHS = (256, 2048, 4096)
MODELS = (("smollm2", "SmolLM2 1.7B Instruct"), ("llama3", "Llama 3 8B Instruct"))
CONTROLS = {"kam_shape_check": ("#166768", "o"), "dot_static": ("#a65025", "D")}
BATCH_STYLE = {1: "-", 8: "--"}


def plot(summary, output):
    data = json.loads(summary.read_text())
    with plt.rc_context({
        "svg.fonttype": "none", "svg.hashsalt": "kam-decode-controls",
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#19242e",
        "axes.labelcolor": "#19242e", "xtick.color": "#324352", "ytick.color": "#324352",
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.35), sharey=True)
        fig.subplots_adjust(left=.1, right=.975, top=.79, bottom=.235, wspace=.16)
        for (name, label), ax in zip(MODELS, axes, strict=True):
            ax.set_title(label, fontweight="bold", pad=10)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e9edf1", linewidth=.8)
            ax.axhline(0, color="#5c6e7d", linewidth=.9)
            ax.set_xticks(range(len(LENGTHS)), [f"{n:,}" for n in LENGTHS])
            ax.tick_params(axis="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for k, (color, marker) in CONTROLS.items():
                for batch, line in BATCH_STYLE.items():
                    values = [data[name][f"{n}x{batch}"]["percent_time_saved"][k] for n in LENGTHS]
                    ax.plot(range(len(values)), values, color=color, marker=marker,
                            linestyle=line, linewidth=1.65, markersize=5.5)
        fig.suptitle("Exact implementation changes on cached requests", x=.1, y=.965,
                     ha="left", fontsize=17, fontweight="bold")
        fig.supylabel("Request time saved (%)", x=.032, y=.51, fontsize=11)
        fig.supxlabel("Input tokens", x=.535, y=.13, fontsize=11)
        markers = [Line2D([0], [0], color=c, marker=m, linewidth=1.5) for c, m in CONTROLS.values()]
        batches = [Line2D([0], [0], color="#566672", linestyle=l) for l in BATCH_STYLE.values()]
        fig.legend(markers+batches, ["KAM skips index scan", "Dot static KV", "Batch 1", "Batch 8"],
                   loc="lower left", bbox_to_anchor=(.085, .018), ncols=4, frameon=False,
                   columnspacing=1.45, handlelength=1.75)
        fig.savefig(output, dpi=180, metadata={"Date": None} if output.suffix == ".svg" else None)
        plt.close(fig)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
