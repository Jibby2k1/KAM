#!/usr/bin/env python3
"""Audit Stage 1 alternating-schedule and factor coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEARNED_GEOMETRIES = {"learned_full", "learned_low_rank_delta"}
FIXED_GEOMETRIES = {"fixed_random", "fixed_data_sample", "fixed_kmeans", "fixed_farthest_point"}
KAM_ARCHITECTURES = {"T-KAM-F", "T-KAM-L"}


def _rows(manifest: Path, run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_root.glob("row_*.json"))]
    return manifest_rows, outputs


def audit(manifest: Path, run_root: Path) -> dict[str, Any]:
    manifest_rows, outputs = _rows(manifest, run_root)
    alternating = [row for row in outputs if str(row.get("optimizer", "")).startswith("alternating_")]
    learned_kam = [row for row in alternating if row.get("architecture") in KAM_ARCHITECTURES and row.get("geometry") in LEARNED_GEOMETRIES]
    fixed_kam = [row for row in alternating if row.get("architecture") in KAM_ARCHITECTURES and row.get("geometry") in FIXED_GEOMETRIES]
    missing_schedule_fields = [str(row.get("row_id")) for row in alternating if not {"alternating_geometry_steps", "alternating_declared_algebra_steps", "alternating_declared_geometry_steps"}.issubset(row.get("metrics", {}))]
    learned_zero = [str(row.get("row_id")) for row in learned_kam if float(row.get("metrics", {}).get("alternating_geometry_steps", 0)) < 1]
    fixed_nonzero = [str(row.get("row_id")) for row in fixed_kam if float(row.get("metrics", {}).get("alternating_geometry_steps", 0)) != 0]
    declared_mismatch: list[str] = []
    for row in alternating:
        parts = str(row.get("optimizer", "")).split("_")
        if len(parts) != 3:
            declared_mismatch.append(str(row.get("row_id")))
            continue
        metrics = row.get("metrics", {})
        if float(metrics.get("alternating_declared_algebra_steps", -1)) != float(parts[1]) or float(metrics.get("alternating_declared_geometry_steps", -1)) != float(parts[2]):
            declared_mismatch.append(str(row.get("row_id")))
    result = {
        "stage_pass": bool(manifest_rows) and len(manifest_rows) == len(outputs) and not missing_schedule_fields and not learned_zero and not fixed_nonzero and not declared_mismatch,
        "manifest_rows": len(manifest_rows),
        "output_rows": len(outputs),
        "task_counts": dict(Counter(str(row.get("task")) for row in outputs)),
        "architecture_counts": dict(Counter(str(row.get("architecture")) for row in outputs)),
        "optimizer_counts": dict(Counter(str(row.get("optimizer")) for row in outputs)),
        "geometry_counts": dict(Counter(str(row.get("geometry")) for row in outputs)),
        "expert_counts": dict(Counter(str(row.get("expert")) for row in outputs)),
        "alternating_rows": len(alternating),
        "learned_geometry_kam_rows": len(learned_kam),
        "learned_geometry_zero_update_row_ids": learned_zero[:100],
        "fixed_geometry_kam_rows": len(fixed_kam),
        "fixed_geometry_nonzero_update_row_ids": fixed_nonzero[:100],
        "missing_schedule_field_row_ids": missing_schedule_fields[:100],
        "declared_schedule_mismatch_row_ids": declared_mismatch[:100],
        "alternating_geometry_step_counts": dict(Counter(float(row.get("metrics", {}).get("alternating_geometry_steps", 0)) for row in alternating)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["stage_pass"] else 1)


if __name__ == "__main__":
    main()
