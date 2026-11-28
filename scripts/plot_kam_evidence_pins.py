#!/usr/bin/env python3
"""Plot paired effects of pinning the marked earlier passage."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BLUE = "#225f96"
TEAL = "#18796f"
LABELS = {"all":"All questions", "linked":"Linked distractor", "unlinked":"Unlinked distractor"}


def plot(source, output):
    summary = json.loads(source.read_text())["panels"]
    fig, axes = plt.subplots(1,2,figsize=(10,4.55),sharex=True,sharey=True,layout="constrained")
    fig.patch.set_facecolor("#f7f8f8")
    fig.suptitle("Direct access helps mainly on unlinked questions at 4K",
                 x=.04, ha="left", fontsize=17, fontweight="bold")
    for ax, width in zip(axes, ("4096","8192"), strict=True):
        ax.set_facecolor("#f7f8f8")
        ax.grid(axis="x", color="#dbe3e6", linewidth=.75)
        ax.axvline(0, color="#51616a", linewidth=1.1, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_title(f"{int(width)//1024}K context", loc="left", fontsize=13, fontweight="bold")
        for i,name in enumerate(LABELS):
            for dataset,dy,color,marker,label in (("musique",.13,TEAL,"o","MuSiQue"),("hotpot",-.13,BLUE,"s","HotpotQA")):
                r = summary[dataset]["widths"][width][name]["paired"]["all-evidence_minus_all-background"]["mirrored"]
                x=100*r["mean_delta"]
                lo,hi=(100*a for a in r["paired_family_bootstrap95_given_seeds"])
                ax.errorbar(x,2-i+dy,xerr=[[x-lo],[hi-x]],color=color,marker=marker,markersize=5,
                            linewidth=1.8,capsize=3,label=label if i==0 else None)
        ax.set_yticks([2,1,0],labels=list(LABELS.values()))
        ax.set_ylim(-.55,2.55)
        ax.set_xlim(-17.5,23)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b8c3c9")
        ax.tick_params(axis="y", length=0, pad=8)
    axes[0].legend(loc="lower left",bbox_to_anchor=(0,-.36),ncol=2,frameon=False)
    axes[1].text(.99,-.28,"Paired 95% intervals by question family",
                 transform=axes[1].transAxes,ha="right",color="#46535b",fontsize=9)
    fig.supxlabel("Change in mirrored retrieval, evidence pins minus background pins (percentage points)",
                   fontsize=11, color="#26373f")
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ("summary","figure"):
        cli.add_argument(k,type=Path)
    a=cli.parse_args()
    plot(a.summary,a.figure)
