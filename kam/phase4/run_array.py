"""Resumable Phase IV manifest-row execution for local or SLURM arrays."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from kam.run_suite import _run_one
from kam.utils import choose_device, save_json

from .table import read_table


def _complete(run_dir: Path) -> bool:
    try:
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")).get("status") == "complete" and (run_dir / "best_model.pt").exists()
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def execute_row(row: dict[str, Any], run_root: str | Path, device_name: str = "auto", resume: bool = True) -> dict[str, Any]:
    run_root = Path(run_root)
    run_dir = run_root / "runs" / str(row["run_id"])
    status_path = run_root / "status" / f"row_{int(row.get('row_id', 0)):06d}.json"
    if resume and _complete(run_dir):
        payload = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "skipped_complete", "path": str(run_dir)}
        save_json(status_path, payload)
        return payload
    start = time.perf_counter()
    try:
        device = choose_device(device_name)
        (run_root / "runs").mkdir(parents=True, exist_ok=True)
        metrics = _run_one(dict(row), run_root / "runs", device, str(row.get("precision", "amp")))
        metrics["phase4_row"] = row
        save_json(run_dir / "metrics.json", metrics)
        payload = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "complete", "path": str(run_dir), "duration_seconds": time.perf_counter() - start, "device": str(device)}
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "failed", "path": str(run_dir), "duration_seconds": time.perf_counter() - start, "error": str(error), "traceback": traceback.format_exc()}
        save_json(run_dir / "failure.json", payload)
    save_json(status_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Phase IV manifest row.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_table(args.manifest)
    index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= index < len(rows):
        raise SystemExit(f"array index {index} outside manifest of {len(rows)} rows")
    result = execute_row(rows[index], args.run_root, args.device, args.resume)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
