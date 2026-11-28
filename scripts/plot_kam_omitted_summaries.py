#!/usr/bin/env python3
"""Plot the paired frozen and live summaries from the pretrained sparse audit."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROWS = ("protected", "mean16", "diagonal16", "mean8", "exact_group_mass16")
LABELS = ("Protected shortlist", "Mean, 16 tokens", "Mean + variance, 16", "Mean, 8 tokens", "Exact mass, 16-token mean")
COLORS = ("#243d57", "#347d89", "#b86742", "#347d89", "#9b5264")


def plot(audit_path, live_path, output_path):
    audit = json.loads(audit_path.read_text())
    live = json.loads(live_path.read_text())
    assert [x["tokens_sha256"] for x in audit["books"]] == [x["tokens_sha256"] for x in live["books"]]
    assert len(live["books"]) == 16 and live["complete"]
    frozen = audit["policies"]
    scores = live["policies"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150,
                         "text.color": "#233143", "axes.labelcolor": "#233143", "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, (first, second) = plt.subplots(1, 2, figsize=(11, 4.75), width_ratios=(1, 1.05))
        figure.subplots_adjust(left=.24, right=.97, bottom=.19, top=.78, wspace=.42)
        figure.text(.045, .93, "What a protected shortlist leaves behind", fontsize=20, weight="bold")
        figure.text(.045, .865, "SmolLM2-135M · 16 disjoint PG-19 test books · 4,096 tokens")
        y = np.arange(len(ROWS))[::-1]
        first.barh(y, [100 * frozen[name]["mean_relative_projected_error"] for name in ROWS],
                   color=COLORS, height=.55, zorder=3)
        first.set_yticks(y, LABELS)
        first.set(xlabel="Error relative to dense energy (%)", xlim=(0, 2.6), ylim=(-.65, 4.65))
        first.set_title("Frozen layer outputs", loc="left", pad=12)
        second.axvline(0, color="#8994a1", linewidth=1, linestyle="--")
        for i, name in enumerate(ROWS[:-1]):
            result = scores[name]
            point = result["delta_vs_protected"]
            low, high = result["paired_book_bootstrap95_vs_protected"]
            second.errorbar(point, y[i], xerr=[[point - low], [high - point]], color=COLORS[i],
                            fmt="o", markersize=6, capsize=3, zorder=3)
        second.text(-.019, .02, "Frozen control only", va="center", color="#697587")
        second.set(yticks=y, yticklabels=[], xlabel="NLL change vs. protected (nats/token)",
                   xlim=(-.023, .026), ylim=(-.65, 4.65))
        second.set_title("Live next-token loss", loc="left", pad=12)
        for axis in (first, second):
            axis.xaxis.grid(color="#e6eaf0")
            axis.tick_params(axis="y", length=0)
            axis.spines["left"].set_visible(False)
        figure.text(.045, .052, "Lower is better in both panels. Live intervals resample books for the fixed checkpoint.",
                    fontsize=9, color="#546275")
        figure.savefig(output_path, dpi=160, facecolor="white")
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "live", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.audit, args.live, args.output)
