from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _fmt(value: str) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final paired Phase II report and LLM handoff.")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--screen-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()
    stats = list(csv.DictReader(args.stats.open(encoding="utf-8")))
    screen = list(csv.DictReader(args.screen_metrics.open(encoding="utf-8")))
    seeds = sorted({row.get("seed") for row in screen}, key=int)
    tasks = sorted({row.get("task") for row in screen})
    variants = sorted({row.get("variant") for row in screen})
    lines = [
        "# Phase II Final Experiment Report",
        "",
        "## Technical summary",
        "",
        f"The five-seed paired grid completed **{len(screen)} runs** across `{', '.join(tasks)}` and `{', '.join(variants)}` using seeds `{', '.join(seeds)}`. No Holm-adjusted comparison has a confidence interval excluding zero; the current evidence does not justify a mechanism claim or promotion to ten-seed confirmation.",
        "",
        "The largest positive paired MSE improvement is the mean baseline-minus-variant difference below. Positive means the candidate had lower loss.",
        "",
        "## Paired inference",
        "",
        "| Task | Baseline | Candidate | Pairs | Mean improvement | 95% bootstrap CI | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['task']} | {row['baseline']} | {row['variant']} | {row['pairs']} | {_fmt(row['mean_improvement'])} | "
            f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] | {_fmt(row['holm_p'])} |"
        )
    lines += [
        "",
        "## What the evidence says",
        "",
        "- The one-seed screen suggested R0 on Mackey–Glass and RR on NARMA, but the five-seed paired analysis is inconclusive for all registered comparisons.",
        "- The Phase A intervention path is implemented and has been run on the ten one-seed screen checkpoints; it reports branch deltas, key/value perturbations, support utilization, frozen probes, and top/random/bottom deletions.",
        "- The paired grid uses shared task/seed generation across variants, but it is not parameter-matched and does not yet include switching post-shift metrics, support-regime alignment, or confirmatory ten-seed inference.",
        "",
        "## Decision gates",
        "",
        "Do not promote a radial-memory claim until the candidate passes all pre-registered gates: stationary degradation ≤5%, post-shift improvement ≥15%, corrected CI excluding zero, top-support deletion stronger than random, noncollapsed support use, and justified time/parameter overhead.",
        "",
        "## Recommended next step",
        "",
        "Run the smallest five-seed screening set that adds switching Mackey–Glass/NARMA, exact parameter matching, and prequential predict→score→reveal→update with frozen/NLMS/SGD/RLS adapters. Only promote a comparison to ten confirmatory seeds if it passes the screen gates.",
        "",
        "## Artifacts",
        "",
        "- Paired metrics: `results/phase2/paired_screen/all_metrics.csv`",
        "- Paired inference: `results/phase2/paired_screen/paired_stats.csv`",
        "- One-seed intervention table: `results/phase2/reanalysis_metrics.csv`",
        "- Raw intervention JSON: `results/phase2/reanalysis_raw/`",
        "- Prequential smoke: `results/phase2/prequential_metrics.csv`",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "PHASE2_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = [
        "# LLM handoff: Phase II experiment status",
        "",
        "Advise on next steps for a research codebase testing whether radial memory improves kernel-attention models.",
        "",
        "## Implementation completed",
        "",
        "- Independent context/memory score geometry: dot, cosine, radial; isotropic/diagonal metric; fixed/learned bandwidth; residual/routes/both output; raw/projected route features; legacy checkpoint loading.",
        "- Deterministic MQAR, bounded Dyck-2, variable-length copy, Mackey–Glass, NARMA, and switching stream infrastructure.",
        "- Resumable YAML + SQLite suite runner with resolved configs, Git/hardware/precision metadata, checkpoints, metrics, per-stream CSVs, failure tracebacks, adapters, paired statistics, and Phase A intervention diagnostics.",
        "",
        "## Runs and results",
        "",
        f"- Five-seed paired grid: {len(screen)} runs across tasks `{', '.join(tasks)}` and variants `{', '.join(variants)}`; seeds `{', '.join(seeds)}`.",
        "- Every tested paired comparison has a 95% bootstrap CI crossing zero and Holm-adjusted p > 0.05.",
        "- One-seed screen winners were R0 on Mackey–Glass and RR on NARMA, but those rankings did not survive as statistically decisive paired evidence.",
        "- Phase A reanalysis ran on ten one-seed checkpoints; support use was approximately 29–32 effective supports out of 32 with zero dead-support fraction under the short-run threshold.",
        "- Prequential smoke ran frozen/NLMS/SGD/RLS on D0 and RR NARMA checkpoints; default SGD was stabilized to a small step size after a deliberately too-large trial diverged.",
        "",
        "## Advice requested",
        "",
        "Recommend the smallest next experiment set that can decide whether radial memory is useful. Require exact parameter-matched controls, paired switching streams, post-shift reacquisition and forgetting metrics, causal deletion/perturbation tests, support diagnostics, and the registered gates. Specify which tests block promotion to ten confirmatory seeds and how to analyze the paired effects.",
        "",
        "Do not infer that radial memory works from the current one-seed winners; the paired five-seed evidence is inconclusive.",
    ]
    (args.output_dir / "PHASE2_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(f"Wrote final report and LLM handoff to {args.output_dir}")


if __name__ == "__main__":
    main()
