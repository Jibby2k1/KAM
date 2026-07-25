"""Stage 2 execution gate checks."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from kam.data.stream_quality import stream_quality_checks


def _generator_name(metrics: dict[str, Any]) -> str:
    metadata = metrics.get("data_metadata", {})
    direct = metadata.get("task_generator") or metadata.get("symbolic_generator")
    if direct:
        return str(direct)
    streams = metadata.get("stream_metadata", {})
    for split in ("train", "validation", "test", "prequential"):
        value = streams.get(split, {}).get("task_generator") if isinstance(streams.get(split), dict) else None
        if value:
            return str(value)
    return ""


def _stable_streams(metrics: dict[str, Any]) -> bool:
    streams = metrics.get("data_metadata", {}).get("stream_metadata", {})
    if not streams:
        return bool(metrics.get("data_metadata", {}).get("symbolic_generator", False))
    for metadata in streams.values():
        quality = metadata.get("stream_quality") if isinstance(metadata, dict) else None
        if not quality or not all(stream_quality_checks(quality).values()):
            return False
    return True


def evaluate_gate(run_root: str | Path, expected: int, *, require_distinct_task_generators: bool = True) -> dict[str, Any]:
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
        "paired_capacity_match": bool(metrics) and all(
            float(row.get("paired_capacity_match_error", 99.0)) <= 0.01
            for row in metrics
        ),
        "stable_controlled_streams": bool(metrics) and all(
            _stable_streams(row) for row in metrics
        ),
        "distinct_task_generators": (not require_distinct_task_generators) or (bool(metrics) and len({_generator_name(row) for row in metrics}) >= 2),
    }
    checks["passed"] = all(checks.values())
    return {"expected": expected, "completed": len(metrics), "failed": len(failure_paths), "checks": checks, "passed": checks["passed"]}
