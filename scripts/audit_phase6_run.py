#!/usr/bin/env python3
"""Audit Phase 6 row outputs against their immutable manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "task",
    "optimizer",
    "architecture",
    "expert",
    "geometry",
    "seed",
    "fidelity",
    "d_model",
    "top_k",
    "num_supports",
)


def _nonfinite(value: Any) -> bool:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(_nonfinite(item) for item in value.values())
    return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(manifest: Path, run_root: Path, expected: int | None = None) -> dict[str, Any]:
    expected_rows = _read_jsonl(manifest)
    output_rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("row_*.json")) + sorted(run_root.glob("row_*.jsonl")):
        output_rows.extend(_read_jsonl(path))
    expected_by_id = {str(row.get("row_id")): row for row in expected_rows}
    output_ids = [str(row.get("row_id")) for row in output_rows]
    output_by_id = {row_id: row for row_id, row in zip(output_ids, output_rows)}
    duplicate_ids = sorted(row_id for row_id, count in Counter(output_ids).items() if count > 1)
    missing_ids = sorted(set(expected_by_id) - set(output_by_id))
    extra_ids = sorted(set(output_by_id) - set(expected_by_id))
    identity_mismatches: list[dict[str, Any]] = []
    nonfinite_ids: list[str] = []
    failure_ids: list[str] = []
    optimizer_mismatches: list[str] = []
    for row_id, expected_row in expected_by_id.items():
        actual = output_by_id.get(row_id)
        if actual is None:
            continue
        differences = {
            field: {"expected": expected_row.get(field), "actual": actual.get(field)}
            for field in IDENTITY_FIELDS
            if actual.get(field) != expected_row.get(field)
        }
        if differences:
            identity_mismatches.append({"row_id": row_id, "differences": differences})
        if actual.get("status") != "pass":
            failure_ids.append(row_id)
        if _nonfinite(actual.get("metrics", {})):
            nonfinite_ids.append(row_id)
        declared_optimizer = expected_row.get("optimizer")
        recorded_optimizer = actual.get("metrics", {}).get("optimizer_mode")
        if declared_optimizer is not None and recorded_optimizer is not None and declared_optimizer != recorded_optimizer:
            optimizer_mismatches.append(row_id)
    expected_count = expected if expected is not None else len(expected_rows)
    stage_pass = (
        len(output_rows) == expected_count
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
        and not identity_mismatches
        and not failure_ids
        and not nonfinite_ids
        and not optimizer_mismatches
    )
    return {
        "stage_pass": stage_pass,
        "manifest": str(manifest),
        "run_root": str(run_root),
        "expected_rows": expected_count,
        "output_rows": len(output_rows),
        "status_counts": dict(Counter(str(row.get("status", "missing")) for row in output_rows)),
        "missing_row_ids": missing_ids[:100],
        "extra_row_ids": extra_ids[:100],
        "duplicate_row_ids": duplicate_ids[:100],
        "identity_mismatches": identity_mismatches[:100],
        "failure_row_ids": failure_ids[:100],
        "nonfinite_row_ids": sorted(set(nonfinite_ids))[:100],
        "optimizer_mismatch_row_ids": optimizer_mismatches[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.manifest, args.run_root, expected=args.expected)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["stage_pass"] else 1)


if __name__ == "__main__":
    main()
