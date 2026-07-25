"""Stage 2 execution gate checks."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def evaluate_gate(run_root: str | Path, expected: int) -> dict[str, Any]:
    root = Path(run_root)
    metric_paths = sorted((root / "runs").glob("*/metrics.json"))
    failure_paths = sorted((root / "runs").glob("*/failure.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    checks = {
        "expected_rows_complete": len(metrics) == expected,
        "no_failure_artifacts": not failure_paths,
        "pilot_checks_passed": bool(metrics) and all(all(row.get("phase5_pilot_checks", {}).values()) for row in metrics),
        "heldout_metrics_present": bool(metrics) and all((path.parent / "heldout_metrics.json").exists() for path in metric_paths),
        "zero_padding": bool(metrics) and all(int(row.get("padding_parameter_count", 0)) == 0 for row in metrics),
        "distinct_task_generators": bool(metrics) and len({str(row.get("data_metadata", {}).get("task_generator", row.get("data_metadata", {}).get("symbolic_generator", ""))) for row in metrics}) >= 2,
    }
    checks["passed"] = all(checks.values())
    return {"expected": expected, "completed": len(metrics), "failed": len(failure_paths), "checks": checks, "passed": checks["passed"]}
