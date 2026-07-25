"""Execute one resumable Stage 1 Phase V pilot row."""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any
from kam.phase4.table import read_table
from kam.run_suite import _run_one
from kam.utils import choose_device, save_json


def _complete(path: Path) -> bool:
    try:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        return metrics.get("status") == "complete" and (path / "best_model.pt").exists()
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def execute_row(row: dict[str, Any], run_root: str | Path, device_name: str = "auto", resume: bool = True) -> dict[str, Any]:
    run_root = Path(run_root)
    run_dir = run_root / "runs" / str(row["run_id"])
    status_path = run_root / "status" / f"row_{int(row.get('row_id', 0)):06d}.json"
    if resume and _complete(run_dir):
        result = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "skipped_complete", "path": str(run_dir)}
        save_json(status_path, result)
        return result
    start = time.perf_counter()
    try:
        device = choose_device(device_name)
        (run_root / "runs").mkdir(parents=True, exist_ok=True)
        metrics = _run_one(dict(row), run_root / "runs", device, str(row.get("precision", "amp")))
        active = int(metrics.get("active_parameter_count", 0))
        target = int(row.get("target_active_parameters", 0))
        match_error = abs(active - target) / max(target, 1)
        pair_match_error = float(
            row.get("paired_capacity_match_error", match_error)
        )
        recorded_count = int(row.get("resolved_active_parameters", active))
        checks = {
            "no_padding_parameters": int(metrics.get("padding_parameter_count", 0)) == 0,
            "active_capacity_match": (
                match_error <= float(row.get("active_match_tolerance", 0.01))
                and pair_match_error
                <= float(row.get("active_match_tolerance", 0.01))
                and active == recorded_count
            ),
            "paired_capacity_match": pair_match_error
            <= float(row.get("active_match_tolerance", 0.01)),
            "best_checkpoint_test_present": metrics.get("best_checkpoint_test") is not None,
            "independent_split_streams": bool(metrics.get("data_metadata", {}).get("independent_split_streams", False)),
            "variant_semantics_recorded": bool(metrics.get("variant_semantics")),
        }
        metrics["active_capacity_match_error"] = match_error
        metrics["paired_capacity_match_error"] = pair_match_error
        metrics["phase5_row"] = row
        metrics["phase5_pilot_checks"] = checks
        save_json(run_dir / "metrics.json", metrics)
        if not all(checks.values()):
            raise RuntimeError(f"Phase V pilot checks failed: {checks}; active={active}, target={target}")
        result = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "complete", "path": str(run_dir), "duration_seconds": time.perf_counter() - start, "device": str(device), "checks": checks}
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "failed", "path": str(run_dir), "duration_seconds": time.perf_counter() - start, "error": str(error), "traceback": traceback.format_exc()}
        save_json(run_dir / "failure.json", result)
    save_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Phase V Stage 1 pilot row.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_table(args.manifest)
    index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    result = execute_row(rows[index], args.run_root, args.device, args.resume)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
