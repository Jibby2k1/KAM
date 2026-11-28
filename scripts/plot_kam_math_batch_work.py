#!/usr/bin/env python3
"""Plot the per-batch decoder lengths from the repetition timing records."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot(archive, destination):
    data = json.loads(gzip.decompress(archive.read_bytes()))
    panels = {output["metadata"]["arm"]: output["records"] for output in data.values()
              if output["metadata"]["batch"] == 8}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#19242e", "axes.labelcolor": "#253a50"}):
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
        figure.subplots_adjust(left=.115, right=.975, top=.775, bottom=.19, wspace=.12)
        for axis, arm, name in zip(axes, ("kam", "dot"), ("KAM", "Dot attention"), strict=True):
            rows = sorted(panels[arm], key=lambda item: item["prompt_tokens"])
            assert len(rows) == 48
            groups = [rows[i:i+8] for i in range(0, len(rows), 8)]
            old = [max(row["original_tokens"] for row in group) for group in groups]
            gated = [max(row["repeat_tokens"] for row in group) for group in groups]
            lengths = [before - after for before, after in zip(old, gated, strict=True)]
            indices = range(1, 7)
            axis.bar(indices, gated, color="#276b69", width=.64)
            saved = [(i, after, length) for i, after, length in zip(indices, gated, lengths, strict=True) if length]
            if saved:
                x_saved, y_saved, steps = zip(*saved, strict=True)
                axis.bar(x_saved, steps, bottom=y_saved, color="#f0d9b3", edgecolor="#785835", width=.64, hatch="///")
            for x, before, after in zip(indices, old, gated, strict=True):
                if before != after:
                    axis.text(x, (before + after) / 2, f"−{before - after}", ha="center", va="center",
                              fontsize=9, bbox={"facecolor": "#fcf4e7", "edgecolor": "none", "pad": 1.5})
            axis.set_title(name, loc="left", fontweight="bold", pad=9)
            axis.set_xticks(indices)
            axis.set_ylim(0, 1080)
            axis.set_yticks((0, 256, 512, 768, 1024))
            axis.grid(axis="y", color="#e6eaf0", linewidth=.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.spines["bottom"].set_color("#bbc8d2")
            axis.tick_params(length=0)
        axes[0].set_ylabel("Longest decode in batch (tokens)")
        figure.suptitle("When an early stop shortens the batch", fontsize=17, fontweight="bold",
                        x=.055, y=.972, ha="left")
        figure.legend(handles=[Patch(color="#276b69", label="With repetition gate"),
                               Patch(facecolor="#f0d9b3", edgecolor="#785835", hatch="///", label="Steps saved")],
                      loc="upper left", bbox_to_anchor=(.045, .91), frameon=False, ncols=2)
        figure.supxlabel("Batch index · Eight requests per batch", y=.075, fontsize=10.5)
        figure.savefig(destination, dpi=170)
        plt.close(figure)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("destination", type=Path)
    args = cli.parse_args()
    plot(args.archive, args.destination)
