#!/usr/bin/env python3
"""Idempotent controller entry point for Phase 6 overnight Slurm jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kam.phase6.overnight_analysis import (
    aggregate_wave,
    preflight_gate,
    stage1_frontier,
    validate_manifest,
)
from kam.phase6.overnight_manifest import build_preflight_rows, build_wave1_rows, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("init", "preflight-gate", "stage1-frontier", "wave1-gate", "wave2-controller", "wave2-gate", "wave3-controller"),
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
