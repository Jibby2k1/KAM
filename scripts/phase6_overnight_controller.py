#!/usr/bin/env python3
"""Idempotent controller entry point for Phase 6 overnight Slurm jobs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kam.phase6.overnight_analysis import (
    aggregate_wave,
    preflight_gate,
    stage1_frontier,
    validate_manifest,
)
from kam.phase6.overnight_manifest import (
    amend_wave1_calibration_fallback_rows,
    amend_wave1_timeout_rows,
    build_preflight_rows,
    build_wave1_rows,
    read_jsonl,
    write_manifest,
)


def _completed_row_ids(rows_root: Path) -> set[str]:
    completed: set[str] = set()
    for path in rows_root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") == "pass":
            completed.add(path.stem)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("init", "repair-wave1", "repair-wave1-calibration", "preflight-gate", "stage1-frontier", "wave1-gate", "wave2-controller", "wave2-gate", "wave3-controller"),
    )
    parser.add_argument("--run-root", default="results/phase6/overnight")
    parser.add_argument("--report-root", default="reports/phase6/overnight")
    parser.add_argument("--stage1-source", default="results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/all_metrics.jsonl")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    manifests = run_root / "manifests"
    if args.action == "init":
        result = {
            "preflight": write_manifest(build_preflight_rows(), manifests / "preflight.jsonl"),
            "wave1": write_manifest(build_wave1_rows(), manifests / "wave1.jsonl"),
        }
    elif args.action == "repair-wave1":
        wave1_path = manifests / "wave1.jsonl"
        backup = manifests / "wave1_pre_timeout_repair.jsonl"
        if not backup.exists():
            shutil.copy2(wave1_path, backup)
        completed_ids = _completed_row_ids(run_root / "rows" / "wave1")
        amended, repair = amend_wave1_timeout_rows(read_jsonl(wave1_path), completed_ids)
        result = {
            "completed_rows_preserved": len(completed_ids),
            "full_manifest": write_manifest(amended, wave1_path),
            "repair_manifest": write_manifest(repair, manifests / "wave1_timeout_repair.jsonl"),
            "superseded_manifest": str(backup),
            "repair_row_ids": [row["row_id"] for row in repair],
        }
    elif args.action == "repair-wave1-calibration":
        wave1_path = manifests / "wave1.jsonl"
        backup = manifests / "wave1_pre_calibration_fallback_repair.jsonl"
        if not backup.exists():
            shutil.copy2(wave1_path, backup)
        completed_ids = _completed_row_ids(run_root / "rows" / "wave1")
        amended, repair = amend_wave1_calibration_fallback_rows(read_jsonl(wave1_path), completed_ids)
        result = {
            "completed_rows_preserved": len(amended) - len(repair),
            "full_manifest": write_manifest(amended, wave1_path),
            "repair_manifest": write_manifest(repair, manifests / "wave1_calibration_fallback_repair.jsonl"),
            "superseded_manifest": str(backup),
            "repair_row_ids": [row["row_id"] for row in repair],
        }
    elif args.action == "preflight-gate":
        result = preflight_gate(run_root)
    elif args.action == "stage1-frontier":
        result = stage1_frontier(Path(args.stage1_source), run_root, report_root)
    elif args.action == "wave1-gate":
        result = aggregate_wave(run_root, "wave1", report_root=report_root)
    elif args.action == "wave2-controller":
        result = validate_manifest(manifests / "wave2.jsonl", 16)
    elif args.action == "wave2-gate":
        result = aggregate_wave(run_root, "wave2", report_root=report_root)
    else:
        result = validate_manifest(manifests / "wave3.jsonl", 8)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
