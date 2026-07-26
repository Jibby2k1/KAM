from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import DEFAULT_TASKS


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    return False


def evaluate_stage0_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "missing")) for row in rows)
    task_counts = Counter(str(row.get("task", "unknown")) for row in rows if row.get("status") == "pass")
    required = set(DEFAULT_TASKS)
    missing = sorted(required - set(task_counts))
    failures = [row.get("row_id", "unknown") for row in rows if row.get("status") != "pass"]
    passed = not missing and not failures and bool(rows)
    return {
        "stage0_pass": passed,
        "row_count": len(rows),
        "status_counts": dict(statuses),
        "passed_task_counts": dict(task_counts),
        "missing_required_tasks": missing,
        "failure_row_ids": failures[:50],
        "large_stage_submission_allowed": passed,
        "interpretation": "Stage 1+ remains blocked until every Stage 0 row passes." if not passed else "Stage 0 validity gate passed; review the report before scaling.",
    }


def evaluate_jsonl(path: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return evaluate_stage0_results(rows)


def evaluate_stage_results(rows: list[dict[str, Any]], *, expected: int | None = None, required_metrics: tuple[str, ...] = ()) -> dict[str, Any]:
    failures = [row.get("row_id", "unknown") for row in rows if row.get("status") != "pass"]
    missing_metrics = []
    for metric in required_metrics:
        if not any(metric in row.get("metrics", {}) for row in rows):
            missing_metrics.append(metric)
    numeric_failures: list[str] = []
    for row in rows:
        if _contains_nonfinite(row.get("metrics", {})):
            numeric_failures.append(str(row.get("row_id", "unknown")))
    row_count_matches_expected = expected is None or len(rows) == expected
    missing_row_count = max(0, expected - len(rows)) if expected is not None else 0
    extra_row_count = max(0, len(rows) - expected) if expected is not None else 0
    passed = bool(rows) and not failures and not missing_metrics and not numeric_failures and row_count_matches_expected
    return {
        "stage_pass": passed,
        "row_count": len(rows),
        "expected": expected,
        "row_count_matches_expected": row_count_matches_expected,
        "missing_row_count": missing_row_count,
        "extra_row_count": extra_row_count,
        "failure_row_ids": failures[:50],
        "missing_metrics": missing_metrics,
        "nonfinite_row_ids": sorted(set(numeric_failures))[:50],
        "large_stage_submission_allowed": passed,
    }
