#!/usr/bin/env python3
"""Plot Flash margins across lengths and observed native QA decision changes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CANVAS="#f7f8f8"
BLUE="#225f96"
TEAL="#18796f"
ORANGE="#a35432"


def plot(source, output):
    p=json.loads(source.read_text())
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    fig.patch.set_facecolor(CANVAS)
    fig.suptitle("Decision margins explain different outcomes at similar logit error",
                 fontweight="bold",fontsize=15,x=.035,ha="left")
    for panel in (ax,bx):
        panel.set_facecolor(CANVAS)
        panel.grid(axis="y",color="#dbe3e6",linewidth=.8)
        panel.set_axisbelow(True)
        panel.spines[["top","right"]].set_visible(False)
        panel.set_ylim(0,100)
    index=np.arange(2)
    for width,color,shift,name in (("4096",BLUE,-.18,"4K"),("8192",TEAL,.18,"8K")):
        low=[]
        for corpus in ("musique","hotpot"):
            data=p[corpus][width]["all"]
            cell=data["policies"]["final-window"]["by_flash_margin"][0]
            low.append(100*cell["n_seed_questions"]/(data["questions"]*3))
        vals=ax.bar(index+shift,low,width=.3,color=color,label=name)
        ax.bar_label(vals,labels=[f"{x:.0f}%" for x in low],padding=2,fontsize=8)
    ax.set_title("Flash near-ties increase with context length",loc="left",fontsize=12,fontweight="bold")
    ax.set_xticks(index,["MuSiQue","HotpotQA"])
    ax.set_ylabel("Flash lead ≤ 0.5 logits (%)")
    ax.legend(frameon=False,loc="upper left")
    all_p=p["quality"]["all"]["policies"]
    for method,color,name in (("protected",BLUE,"Protected"),("output-selected",ORANGE,"Output-selected"),
                               ("final-window",TEAL,"Final Flash window")):
        rates=[100*m["fraction_changed"] for m in all_p[method]["by_flash_margin"]]
        bx.plot(range(4),rates,linewidth=2,marker="o",color=color,label=name)
    bx.set_title("QuALITY decisions changed by policy",loc="left",fontsize=12,fontweight="bold")
    bx.set_xticks(range(4),["≤ 0.5","0.5–1","1–2","> 2"])
    bx.set_xlabel("Flash lead over runner-up (logits)")
    bx.set_ylabel("Changed decisions (%)")
    bx.legend(frameon=False,fontsize=8)
    fig.savefig(output,dpi=180,facecolor=CANVAS,bbox_inches="tight")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("result",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    plot(a.result,a.output)
