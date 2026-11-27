#!/usr/bin/env python3
"""Render the overnight research evidence panels from archived summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#152632"
MUTE = "#526777"
GRID = "#e2e9ec"
BG = "#f8faf9"
BLUE = "#275d88"
TEAL = "#007a73"
ORANGE = "#b0610e"
PALE = "#8aa9bc"


def read(root, folder, file="summary.json"):
    return json.loads((root / folder / file).read_text())


def setup(ax, title, details):
    ax.set_facecolor(BG)
    ax.text(0, 1.20, title, transform=ax.transAxes, ha="left", va="bottom",
            color=INK, fontweight="bold", fontsize=15)
    ax.text(0, 1.12, details, transform=ax.transAxes, ha="left", va="bottom", color=MUTE, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelcolor=MUTE, labelsize=8, pad=5)
    ax.set_axisbelow(True)


def hotpot(ax, archive):
    setup(ax, "HotpotQA", "Change after adding conditioned search · 1,024 questions")
    panel = archive["cohorts"]
    labels = ["Complete pool\n640 questions", "Incomplete pool\n384 questions"]
    y = np.arange(2)
    for shift, metric, color, name in [(-0.17, "support", BLUE, "Both supports"),
                                       (0.17, "joint_exact", TEAL, "Joint exact match")]:
        value = [panel[k]["stages"]["conditioned_union"][metric] - panel[k]["stages"]["broad"][metric]
                 for k in ("complete_original", "incomplete_original")]
        ax.barh(y + shift, value, height=0.25, color=color, label=name)
        for pos, v in zip(y, value, strict=True):
            ax.text(v + (1.1 if v >= 0 else -1.1), pos + shift, f"{v:+}", va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax.axvline(0, color=MUTE, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-25, 54)
    ax.set_ylim(1.55, -0.48)
    ax.set_xlabel("Change in question count", fontsize=8, color=MUTE)
    ax.set_xticks([-20, 0, 20, 40])
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="center right", frameon=False, fontsize=8, labelcolor=MUTE)


def twowiki(ax, archive, refit):
    setup(ax, "2WikiMultiHopQA", "Evidence and accuracy at 128 candidates · 1,024 questions")
    ready = archive["groups"]["all"]["counts"]
    fixed = refit["scopes"]["all"]["counts"]["refit"]
    x = np.arange(3)
    arms = [("Dense", ready["dense128"], "o", PALE, -55),
            ("Conditioned / refit reranker", fixed, "s", ORANGE, 30),
            ("Conditioned / frozen reranker", ready["conditioned128"], "D", TEAL, -46)]
    for label, r, marker, color, dy in arms:
        scores = [r[k] for k in ("pool", "support", "joint")]
        ax.plot(x, scores, marker=marker, markersize=5, label=label, linewidth=2, color=color, alpha=0.92)
        for i, v in zip(x, scores, strict=True):
            if i == 0 and label.startswith("Conditioned / refit"):
                continue
            text_x = i
            if i == 2:
                if "frozen" in label:
                    offset = 48
                elif "refit" in label:
                    text_x, offset = i + 0.13, -57
                else:
                    text_x, offset = i - 0.13, -49
            else:
                offset = 48 if i == 0 and "frozen" in label else dy
            ax.text(text_x, v + offset, str(v), color=INK, ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.24, 2.28)
    ax.set_ylim(0, 1045)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_xticks(x, ["Both articles\nin pool", "Both articles\nselected", "Joint exact\nmatch"])
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, labelcolor=MUTE)


def musique(ax, learned, bindings):
    setup(ax, "MuSiQue", "Complete chain recovery · four-hop slice of 160")
    items = learned["scopes"]["4"]
    ladder = bindings["scopes"]["4"]["metrics"]
    generated = [seed["counts"]["learned"]["full_support"] for seed in items.values()]
    numbers = [next(iter(items.values()))["counts"]["hybrid"]["full_support"],
               np.median(generated), ladder["top1"]["full_support"],
               ladder["gold_paragraph"]["full_support"], ladder["resolved_answers"]["full_support"]]
    y = np.arange(5)
    ax.barh(y, numbers, height=0.55, color=[PALE, TEAL, ORANGE, ORANGE, ORANGE])
    labels = ["Hybrid", "Learned decomposition", "Author steps + predicted answers",
              "Author steps + gold paragraphs", "Author steps + gold answers"]
    ax.set_yticks(y, labels)
    for i, value in enumerate(numbers):
        mark = f"{min(generated)}–{max(generated)}" if i == 1 else str(value)
        end = max(generated) if i == 1 else value
        ax.text(end + 2, i, mark, ha="left", va="center", color=INK, fontsize=8)
    ax.hlines(1, min(generated), max(generated), color=INK, linewidth=1.4)
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Questions with all gold paragraphs", color=MUTE, fontsize=8)
    ax.grid(axis="x", color=GRID)


def latency(ax, capture):
    setup(ax, "KAM decoder training", "19,200 updates from scratch · times include capture/setup")
    labels, pairs = [], []
    for key, label in (("l4", "L4"), ("b200", "B200"), ("rtx6000", "RTX PRO")):
        for arm in ("dot", "radial"):
            labels.append(f"{label} {arm}")
            t = capture[key]["arms"][arm]["times"]
            pairs.append(tuple(t[k]["incl_setup_seconds"] for k in ("eager", "shape_graph")))
    y = np.arange(len(pairs))
    for idx, (base, candidate) in enumerate(pairs):
        ax.barh(idx - .15, base, height=.26, color=PALE)
        ax.barh(idx + .15, candidate, height=.26, color=TEAL)
        for offset, v in ((-.15, base), (.15, candidate)):
            ax.text(v + 2.3, idx + offset, f"{v:.1f} s", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 183)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.grid(axis="x", color=GRID)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALE, label="Eager"),
                       plt.Rectangle((0, 0), 1, 1, color=TEAL, label="Per-shape CUDA Graphs")],
              loc="upper right", frameon=False, labelcolor=MUTE, fontsize=8)


def archive_transfer(ax, record):
    setup(ax, "Temporal rank with raw timestamps", "Strict key and snapshot accuracy · 48-record banks")
    selected = record["outcomes"]["dot"]["real"]
    x = np.arange(3)
    domains = ("dallas", "unemployment", "industry")
    for move, arm, label, color in (
        (-.17, "reference", "Timestamp offsets", BLUE),
        (+.17, "fact_rank_feature", "Offsets and within-address rank", TEAL),
    ):
        rows = [selected[d]["48"][arm] for d in domains]
        values = [row["counts"]["joint"]/row["n"]*100 for row in rows]
        ax.bar(x + move, values, width=.28, color=color, label=label)
        for i, score in zip(x, values, strict=True):
            ax.text(i+move, score+2.2, f"{score:.1f}%", va="bottom", ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, ["Dallas GDP", "Eurostat\nunemployment", "Eurostat\nindustrial output"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left", frameon=False, fontsize=7.3, labelcolor=MUTE)


def queued_tail(ax, queue):
    setup(ax, "HotpotQA under GPU queueing", "Median shard p95 · four matched FIFO arrival traces")
    variants = (("fp16_width8", "FP16, cap 8", BLUE, "o"), ("fp16_width16", "FP16, cap 16", TEAL, "s"),
                ("int8_width16", "INT8, cap 16", ORANGE, "^"))
    rates = sorted(queue["trials"], key=int)
    for key, label, color, marker in variants:
        rows = [queue["trials"][rate][key] for rate in rates]
        x = [r["median_shard_offered_qps"] for r in rows]
        times = [r["median_shard_p95_ms"]["end_to_end_ms"] for r in rows]
        ax.plot(x, times, color=color, label=label, marker=marker, linewidth=2, markersize=5)
    ax.set_xticks([12, 24, 30])
    ax.set_xlim(10.8, 30.5)
    ax.set_ylim(0, 1500)
    ax.set_ylabel("Arrival-to-answer latency (ms)", color=MUTE, fontsize=9)
    ax.set_xlabel("Offered requests per second", color=MUTE, fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper left", frameon=False, labelcolor=MUTE, fontsize=8)


def transfer_and_serving(root, output):
    fig = plt.figure(figsize=(11.2, 4.95))
    grid = fig.add_gridspec(1, 2, left=.083, bottom=.20, right=.95, top=.67, wspace=.26)
    fig.text(.083, .939, "Transfer and serving conditions", fontsize=24, color=INK, fontweight="bold", va="top")
    archive_transfer(fig.add_subplot(grid[0, 0]), read(root, "address_local_clock_v2", "summary-53006700.json"))
    queued_tail(fig.add_subplot(grid[0, 1]), read(root, "fullwiki_live_queue_v3"))
    fig.text(.083, .057, "Temporal arms use typed addresses, matched mixed dot training and three seeds. All serving policies returned the same outcomes.",
             color=MUTE, fontsize=8.3)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def routing_and_training(root, output):
    routes = read(root, "musique_endpoint_routing_v1")["quality"]["2"]
    schedules = read(root, "typed_guarded_sdpa_curriculum_v2")["devices"]["NVIDIA L4"]["radial"]["outcomes"]
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(1, 2, left=.20, right=.96, bottom=.19, top=.72, wspace=.62)
    fig.text(.073, .935, "Evidence routing and training order", fontsize=24, weight="bold", color=INK, va="top")
    ax = fig.add_subplot(grid[0, 0])
    setup(ax, "Natural two-hop endpoints", "Joint exact match · 336 presentations across both orders")
    policies = ("true_prefix", "true_prefix_route_masked", "true_prefix_route_endpoint_only", "gold_route_endpoint_only")
    ticks = np.arange(len(policies))
    for name, shift, color in (("base", -.16, BLUE), ("large", .16, TEAL)):
        items = [routes["reader"][name][mode] for mode in policies]
        values = [v.get("joint", v["answers"])["correct"] for v in items]
        ax.barh(ticks+shift, values, height=.26, label=f"FLAN-T5-{name}", color=color)
        for pos, value in zip(ticks, values, strict=True):
            ax.text(value+4, pos+shift, str(value), color=INK, va="center", fontsize=8)
    ax.set_yticks(ticks, ["Both endpoints", "Selected, rival masked", "Selected endpoint only", "Gold endpoint only"])
    ax.invert_yaxis()
    ax.set_xlim(0, 265)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Presentations with correct evidence and answer", fontsize=8, color=MUTE)
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.18), ncols=2, frameon=False, fontsize=8, labelcolor=MUTE)

    ax = fig.add_subplot(grid[0, 1])
    setup(ax, "Guarded radial training", "New depth-1,024 correction episodes · 1,536 across seeds")
    control = ("interleaved", "shuffled_no_warmup", "evenly_warmed_400", "evenly_warmed_800", "bookends_400")
    positions = np.arange(len(control))
    counts = [schedules[key]["synthetic"]["fresh/corrections/1024"]["correct"] for key in control]
    ax.barh(positions, counts, height=.58, color=[PALE, PALE, TEAL, TEAL, ORANGE])
    ax.set_yticks(positions, ["Interleaved", "Shuffled", "400 shallow first", "800 shallow first", "400 shallow at each end"])
    ax.invert_yaxis()
    for pos, value in zip(positions, counts, strict=True):
        ax.text(value+25, pos, f"{value:,}", color=INK, va="center", fontsize=8)
    ax.set_xlim(0, 1750)
    ax.set_xticks([0, 400, 800, 1200, 1600])
    ax.set_xlabel("Episodes with exact alias and snapshot", fontsize=8, color=MUTE)
    ax.grid(axis="x", color=GRID)
    fig.text(.073, .048, "Verified earlier evidence is supplied to the endpoint selector. All training schedules use the same batches and token budget.",
             color=MUTE, fontsize=8.2)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--transfer-and-serving", type=Path)
    p.add_argument("--routing-and-training", type=Path)
    a = p.parse_args()
    d = a.results
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "savefig.facecolor": BG})
    fig = plt.figure(figsize=(13, 8.7))
    shape = fig.add_gridspec(2, 2, left=0.22, bottom=0.14, right=0.95, top=0.77, hspace=0.78, wspace=0.47)
    fig.text(0.072, 0.931, "Overnight KAM evidence", fontsize=28, color=INK, fontweight="bold")
    fig.text(0.073, 0.886, "Natural retrieval transfer and hardware measurements  ·  November 27, 2026",
             color=MUTE, fontsize=12)
    hotpot(fig.add_subplot(shape[0, 0]), read(d, "conditioned_union_v1"))
    twowiki(fig.add_subplot(shape[0, 1]), read(d, "native_twowiki_rerank_v1"), read(d, "native_twowiki_refit_v1"))
    musique(fig.add_subplot(shape[1, 0]), read(d, "musique_learned_decomposition_v1"), read(d, "musique_bindings_v1"))
    latency(fig.add_subplot(shape[1, 1]), read(d, "curriculum_capture_v1", "summary-three-gpus.json"))
    fig.text(0.072, 0.055, "Orange marks MuSiQue controls that receive author information. Training timings exclude data setup and evaluation.",
             fontsize=9, color=MUTE)
    fig.savefig(a.output, dpi=165)
    plt.close(fig)
    if a.transfer_and_serving:
        transfer_and_serving(d, a.transfer_and_serving)
    if a.routing_and_training:
        routing_and_training(d, a.routing_and_training)


if __name__ == "__main__":
    main()
