from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gates import evaluate_stage0_results
from .report import build_stage0_report
from .gates import evaluate_stage_results
from .plots import plot_learning_curves, plot_memory_diagnostics, plot_prediction_true_error, plot_router_load
from .artifacts import write_artifacts


def aggregate_stage0(run_root: str | Path, expected: int, report_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    files = sorted(root.glob("row_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"expected one row in {path}, found {len(lines)}")
        rows.append(json.loads(lines[0]))
    rows.sort(key=lambda row: str(row.get("row_id", "")))
    combined = root / "validity_results.jsonl"
    combined.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    artifacts = write_artifacts(rows, root)
    gate = evaluate_stage0_results(rows)
    if len(rows) != expected:
        gate = {
            **gate,
            "stage0_pass": False,
            "large_stage_submission_allowed": False,
            "missing_row_count": expected - len(rows),
            "interpretation": "Stage 0 is blocked because the aggregate did not receive every expected row output.",
        }
    report_path = Path(report_root) / "PHASE6_STAGE0_VALIDITY_REPORT_HPG.md"
    build_stage0_report(combined, report_path, execution="hpg")
    (Path(report_root) / "PHASE6_STAGE0_GATE_HPG.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(rows), **gate, "combined": str(combined), "report": str(report_path)}


def _read_row_outputs(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("row_*.json")) + sorted(run_root.glob("row_*.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            rows.append(json.loads(line))
    return rows


def aggregate_stage(run_root: str | Path, *, expected: int | None = None, report_root: str | Path = "reports/phase6", stage: str = "stage") -> dict[str, Any]:
    """Aggregate static row outputs without shared SQLite state."""
    root = Path(run_root)
    rows = _read_row_outputs(root)
    rows.sort(key=lambda row: str(row.get("row_id", "")))
    combined = root / "all_metrics.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    artifacts = write_artifacts(rows, root)
    gate = evaluate_stage_results(rows, expected=expected)
    destination = Path(report_root)
    destination.mkdir(parents=True, exist_ok=True)
    numeric_fields = sorted({key for row in rows for key, value in row.get("metrics", {}).items() if isinstance(value, (float, int))})
    metric_summary = {}
    for field in numeric_fields:
        values = [float(row["metrics"][field]) for row in rows if field in row.get("metrics", {})]
        if values:
            metric_summary[field] = {"n": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    (root / "metrics_summary.json").write_text(json.dumps(metric_summary, indent=2) + "\n", encoding="utf-8")
    histories = [row.get("metrics", {}).get("loss_history") for row in rows]
    histories = [list(map(float, history)) for history in histories if isinstance(history, list) and history]
    if histories:
        width = max(len(history) for history in histories)
        mean_curve = [sum(history[index] for history in histories if index < len(history)) / sum(index < len(history) for history in histories) for index in range(width)]
        plot_learning_curves({"mean_training_loss": mean_curve}, root / "learning_curves.png")
    else:
        losses = [float(row["metrics"]["loss"]) for row in rows if "loss" in row.get("metrics", {})]
        if losses:
            plot_learning_curves({"row_loss_fallback": losses}, root / "learning_curves.png")
    entropy = [float(row["metrics"][key]) for row in rows for key in row.get("metrics", {}) if key.endswith("routing_entropy")]
    effective = [float(row["metrics"][key]) for row in rows for key in row.get("metrics", {}) if key.endswith("effective_support_count")]
    dead = [float(row["metrics"][key]) for row in rows for key in row.get("metrics", {}) if key.endswith("dead_support_fraction")]
    balance = [float(row["metrics"][key]) for row in rows for key in row.get("metrics", {}) if key.endswith("load_balance_error")]
    if entropy or effective or dead or balance:
        plot_memory_diagnostics({"routing_entropy": entropy, "effective_support_count": effective, "dead_support_fraction": dead, "load_balance_error": balance}, root / "memory_diagnostics.png")
    loads = [float(row["metrics"][key]) for row in rows for key in row.get("metrics", {}) if key.endswith("tokens_per_support")]
    if loads:
        plot_router_load(loads, root / "router_load.png")
    summary = {"stage": stage, "rows": len(rows), **gate, "combined": str(combined), "artifacts": artifacts, "metric_summary": str(root / "metrics_summary.json")}
    (destination / f"{stage}_aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = destination / f"{stage}_report.md"
    report.write_text(
        "# Phase 6 {stage} report\n\n".format(stage=stage)
        + f"- Rows: **{len(rows)}**\n- Expected rows: **{gate.get('expected', 'unspecified')}**\n- Missing expected rows: **{gate.get('missing_row_count', 0)}**\n- Gate: **{'PASS' if gate['stage_pass'] else 'BLOCKED'}**\n- Failures: **{len(gate['failure_row_ids'])}**\n\n"
        + "This report summarizes executable row outputs. It is not confirmatory evidence unless the declared upstream gates and statistical requirements are satisfied.\n",
        encoding="utf-8",
    )
    return {**summary, "report": str(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 6 Stage 0 HPG row outputs")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--stage", default="stage0_validity")
    parser.add_argument("--generic", action="store_true")
    args = parser.parse_args()
    result = aggregate_stage(args.run_root, expected=args.expected, report_root=args.report_root, stage=args.stage) if args.generic else aggregate_stage0(args.run_root, args.expected, args.report_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
