#!/usr/bin/env python3
"""Plot marginal mass versus reconstruction and leave-one-book-out error."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE="#225f96"
TEAL="#18796f"
LIGHT="#f7f8f8"


def plot(source,output):
    r=json.loads(source.read_text())
    x=100*np.array(r["layer_marginals"]["retained_mass_gain"])
    y=100*np.array(r["layer_marginals"]["output_error_drop"])
    chosen=np.array(r["all_calibration_schedules"]["output-four-twelve"]["schedule"])==12
    figure,axes=plt.subplots(1,2,figsize=(10,4.45),layout="constrained")
    figure.patch.set_facecolor(LIGHT)
    figure.suptitle("Retained mass and output fidelity favor different layers",
                    ha="left",x=.035,fontweight="bold",fontsize=16)
    ax,bx=axes
    for panel in axes:
        panel.set_facecolor(LIGHT)
        panel.grid(color="#dfe6e9",linewidth=.7)
        panel.set_axisbelow(True)
        panel.spines[["top","right"]].set_visible(False)
    ax.set_title("Marginal change from 4 to 12 blocks",loc="left",fontsize=12,fontweight="bold")
    ax.scatter(x[~chosen],y[~chosen],s=37,color=BLUE,label="Other layers")
    ax.scatter(x[chosen],y[chosen],s=38,color=TEAL,label="Output-selected layers")
    for i in sorted({int(x.argmax()),int(y.argmax()),int(x.argmin())}):
        ax.annotate(f"Layer {i+1}",(x[i],y[i]),xytext=(5,7),textcoords="offset points",fontsize=8,color="#34434b")
    ax.axhline(0,linestyle="--",color="#788990",linewidth=1)
    ax.set_xlabel("More dense mass retained (percentage points)")
    ax.set_ylabel("Drop in relative squared output error (points)")
    ax.legend(loc="upper right",frameon=False,fontsize=8)
    bx.set_title("Reconstruction on each held-out book",loc="left",fontsize=12,fontweight="bold")
    panels=list(r["per_book"].values())
    old=100*np.array([t["uniform-eight"]["scores"]["output_error"] for t in panels])
    new=100*np.array([t["output-four-eight-twelve"]["scores"]["output_error"] for t in panels])
    for a,b in zip(old,new,strict=True):
        bx.plot([0,1],[a,b],linewidth=.9,color="#99aeb6",alpha=.85)
    bx.scatter(np.zeros_like(old),old,c=BLUE,s=27,zorder=3)
    bx.scatter(np.ones_like(new),new,c=TEAL,s=27,zorder=3)
    bx.set_xticks([0,1],["Uniform eight","Selected 4, 8, or 12"])
    bx.set_xlim(-.25,1.25)
    bx.set_ylabel("Relative squared output error (%)")
    figure.savefig(output,dpi=180,facecolor=LIGHT,bbox_inches="tight")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for a in ("summary","output"):
        parser.add_argument(a,type=Path)
    x=parser.parse_args()
    plot(x.summary,x.output)
