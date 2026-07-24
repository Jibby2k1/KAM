from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _load_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in csv.DictReader(metrics_path.open(encoding="utf-8"))]
    run_root = metrics_path.parent
    for row in rows:
        run_metrics = run_root / str(row.get("run_id", "")) / "metrics.json"
        if run_metrics.exists():
            payload = json.loads(run_metrics.read_text(encoding="utf-8"))
            row.update({f"final_{key}": value for key, value in payload.get("final_validation", {}).items()})
            row.update({f"best_{key}": value for key, value in payload.get("best_validation", {}).items()})
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in csv.DictReader(path.open(encoding="utf-8"))] if path.exists() else []


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _screen_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for task in sorted({str(row.get("task", "unknown")) for row in rows}):
        candidates = [row for row in rows if row.get("task") == task]
        mse_candidates = [row for row in candidates if _number(row.get("final_mse")) is not None]
        mae_candidates = [row for row in candidates if _number(row.get("final_mae")) is not None]
        if not mse_candidates:
            continue
        best_mse = min(mse_candidates, key=lambda row: float(row["final_mse"]))
        best_mae = min(mae_candidates, key=lambda row: float(row["final_mae"])) if mae_candidates else None
        lines.append(
            f"- **{task}:** lowest observed MSE was **{_fmt(_number(best_mse.get('final_mse')))}** "
            f"({best_mse.get('variant')}); lowest MAE was **{_fmt(_number(best_mae.get('final_mae')) if best_mae else None)}** "
            f"({best_mae.get('variant') if best_mae else '—'})."
        )
    return lines


def _screen_table(rows: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("task", "unknown")), str(row.get("variant", "unknown")))].append(row)
    lines = ["| Task | Variant | n | Final MSE | Final MAE | Params | Time (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for (task, variant), group in sorted(groups.items()):
        lines.append(
            f"| {task} | {variant} | {len(group)} | {_fmt(_avg(group, 'final_mse'))} | {_fmt(_avg(group, 'final_mae'))} | "
            f"{_fmt(_avg(group, 'parameter_count'))} | {_fmt(_avg(group, 'total_seconds'))} |"
        )
    return lines


def _intervention_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Task | Variant | Baseline loss | Context Δ | Memory Δ | Key Δ | Value Δ | Effective supports | Probe MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item.get("task")), str(item.get("model")))):
        lines.append(
            f"| {row.get('task', '—')} | {row.get('model', '—')} | {_fmt(_number(row.get('baseline_loss')))} | "
            f"{_fmt(_number(row.get('context_ablation_delta')))} | {_fmt(_number(row.get('memory_ablation_delta')))} | "
            f"{_fmt(_number(row.get('key_perturb_delta')))} | {_fmt(_number(row.get('value_perturb_delta')))} | "
            f"{_fmt(_number(row.get('global_effective_supports')))} | {_fmt(_number(row.get('probe_mse')))} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase II evidence and LLM handoff reports.")
    parser.add_argument("--metrics", type=Path, default=Path("results/phase2/dynamic_screen/all_metrics.csv"))
    parser.add_argument("--reanalysis", type=Path, default=Path("results/phase2/reanalysis_metrics.csv"))
    parser.add_argument("--inventory", type=Path, default=Path("results/phase2/checkpoint_inventory.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()

    screen_rows = [row for row in _load_rows(args.metrics) if row.get("status") == "complete"]
    intervention_rows = [row for row in _load_csv(args.reanalysis) if row.get("baseline_loss") not in (None, "")]
    tasks = sorted({str(row.get("task", "unknown")) for row in screen_rows})
    variants = sorted({str(row.get("variant", "unknown")) for row in screen_rows})
    inventory_count = len(_load_csv(args.inventory))
    lines = [
        "# Phase II Experiment Results",
        "",
        "## Technical summary",
        "",
        f"The bounded GPU screen completed **{len(screen_rows)} runs** across **{len(tasks)} tasks** and variants `{', '.join(variants)}`. It is useful for shortlisting confirmatory comparisons, but every screen comparison has one seed and therefore cannot establish statistical significance or generalization.",
        "",
        "Observed screen winners by task:",
        *_screen_summary(screen_rows),
        "",
        f"Phase A intervention reanalysis completed on **{len(intervention_rows)} screen checkpoints**; the broader checkpoint inventory contains **{inventory_count} rows**. The interventions are diagnostic evidence, not a registered confirmatory result.",
        "",
        "## Quantitative screen evidence",
        "",
        "Validation metrics are loader averages at the end of training; lower MSE/MAE is better. `n` is the number of completed runs in each task/variant group.",
        "",
        *_screen_table(screen_rows),
        "",
        "## Intervention evidence: the memory path is measurable, but not yet validated across seeds",
        "",
        "For each available memory model, Δ is the validation-loss change relative to the intact checkpoint after one intervention. Positive deletion/ablation deltas indicate higher loss. The reanalysis used up to four validation batches and saves raw loss vectors plus deletion curves alongside the CSV.",
        "",
        *_intervention_table(intervention_rows),
        "",
        "The support diagnostics show roughly 29–32 effective supports out of 32 for the screen models, with zero dead-support fraction under the configured threshold. That is a healthy noncollapse signal for this short run, not evidence of stable support utilization across training or seeds.",
        "",
        "## Scope and method",
        "",
        f"- Tasks: `{', '.join(tasks)}`; variants: `{', '.join(variants)}`; device: CUDA AMP; seeds: `{sorted({row.get('seed') for row in screen_rows})}`.",
        "- The implementation supports independent context/memory score modes, dot/cosine/radial scores, isotropic/diagonal radial metrics, fixed/learned bandwidth, residual/routes/both memory output, raw/projected route features, and legacy checkpoint loading.",
        "- Each run persists resolved configuration, Git metadata when available, hardware/precision, parameter counts, training steps/samples, elapsed time, peak GPU memory, checkpoint, metrics, stream CSV, and failure tracebacks.",
        "- Reanalysis includes context and memory branch ablations, key/value noise perturbations, global support-utilization diagnostics, frozen ridge probes, top/random/bottom support deletion curves, and raw JSON diagnostics.",
        "",
        "## Limitations and quality checks",
        "",
        "- One seed per task/variant means paired bootstrap confidence intervals, sign-permutation p-values, Holm correction, and cross-seed stability are not estimable yet.",
        "- Screen streams were independently generated. Confirmatory runs must fix data seeds and use paired streams across variants.",
        "- Stationary-degradation, post-shift reacquisition, forgetting, regime alignment, and causal-deletion gates are not met by this screen. The deletion intervention is present, but its evidence is single-seed and short-horizon.",
        "- The flat manifest is metadata; the report joins nested `metrics.json` validation fields before interpreting model performance.",
        "",
        "## Recommended next steps",
        "",
        "1. Run five screening seeds on paired MQAR, switching NARMA, and switching Mackey–Glass streams using D0/R0/DD/DR/RR plus exact parameter-matched controls.",
        "2. Pre-register the primary comparison and gates: stationary degradation ≤5%, post-shift improvement ≥15%, confidence interval excluding zero, top-support deletion stronger than random, noncollapsed supports, and justified overhead.",
        "3. Promote only comparisons that pass screening to ten confirmatory seeds, then apply paired bootstrap/permutation and Holm correction.",
        "4. Keep conditional TinyStories disabled until the dynamic-memory gate passes.",
        "",
        "## Further questions",
        "",
        "- Does radial memory remain useful after exact parameter matching and fixed paired streams?",
        "- Does it improve post-shift reacquisition without exceeding the stationary-degradation gate?",
        "- Are learned bandwidths and support assignments stable across seeds, or are the screen differences optimization noise?",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "PHASE2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    handoff = [
        "# LLM handoff: Phase II experiment status",
        "",
        "You are advising on next steps for a research codebase extending kernel-attention experiments.",
        "",
        "## Implemented",
        "",
        "- Independent context/memory score architecture: dot, cosine, radial; isotropic/diagonal radial metric; fixed/learned bandwidth; residual/routes/both memory output; raw/projected route features.",
        "- MQAR, bounded Dyck-2, variable-length-capable task infrastructure, Mackey–Glass/NARMA generators and switching schedules; resumable YAML suites with SQLite registry, checkpoints, metrics, hardware metadata and failures.",
        "- Phase A interventions: branch ablation, key/value perturbation, support utilization, frozen ridge probe, and top/random/bottom support deletion curves with raw JSON diagnostics.",
        "",
        "## Completed evidence",
        "",
        f"- GPU screen: {len(screen_rows)} runs, tasks `{', '.join(tasks)}`, variants `{', '.join(variants)}`, one seed `{sorted({row.get('seed') for row in screen_rows})}`.",
        *_screen_summary(screen_rows),
        f"- Intervention reanalysis: {len(intervention_rows)} checkpoints; checkpoint inventory: {inventory_count} rows.",
        "- Effective support counts were approximately 29–32/32 with zero dead-support fraction in the short screen reanalysis.",
        "",
        "## Interpretation",
        "",
        "The evidence is enough to choose confirmatory comparisons, not to claim that radial memory is the cause of the observed task differences. The memory intervention path is now implemented, but all evidence is one seed and the dynamic-shift gates are still untested.",
        "",
        "## Advice requested",
        "",
        "Recommend the smallest next experiment set that can decide whether radial memory is useful. Prioritize paired data seeds, exact parameter-matched controls, switching streams, causal deletion/perturbation tests, support diagnostics, and the registered gates. State which tests should block confirmatory claims and how to allocate five screening seeds versus ten confirmatory seeds.",
    ]
    (args.output_dir / "PHASE2_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    (args.output_dir / "PHASE2_DECISION_MEMO.md").write_text(
        "# Decision memo\n\n" + "\n".join(_screen_summary(screen_rows)) + "\n\n" +
        f"Intervention reanalysis covers {len(intervention_rows)} checkpoints. Treat this as shortlist evidence only; confirmatory claims require paired streams, multiple seeds, preregistered gates, and corrected inference.\n",
        encoding="utf-8",
    )
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
