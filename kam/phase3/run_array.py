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

from .evaluate import causal_probe, prequential_evaluate
from .manifest import load_manifest


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _is_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metrics.get("status") == "complete"


def execute_row(row: dict[str, Any], run_root: str | Path, *, device_name: str = "auto", resume: bool = True) -> dict[str, Any]:
    run_root = Path(run_root)
    runs_root = run_root / "runs"
    run_dir = runs_root / str(row["run_id"])
    status_path = run_root / "status" / f"row_{int(row.get('row_id', 0)):06d}.json"
    if resume and _is_complete(run_dir):
        payload = {"row_id": row.get("row_id"), "run_id": row.get("run_id"), "status": "skipped_complete", "path": str(run_dir)}
        _write_status(status_path, payload)
        return payload
    start = time.perf_counter()
    device = choose_device(device_name)
    row = dict(row)
    precision = str(row.get("precision", "amp"))
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        metrics = _run_one(row, runs_root, device, precision)
        checkpoint_name = "final_model.pt" if bool(row.get("use_final_checkpoint_for_diagnostics", False)) else "best_model.pt"
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "best_model.pt"
        extra: dict[str, Any] = {"diagnostic_checkpoint": str(checkpoint_path)}
        if bool(row.get("online_eval", False)):
            extra["prequential"] = prequential_evaluate(checkpoint_path, row, run_dir, device)
        if bool(row.get("causal_probe", False)):
            extra["causal_probe"] = causal_probe(checkpoint_path, row, run_dir, device)
        metrics.update({"phase3": extra, "phase3_row": row})
        save_json(run_dir / "metrics.json", metrics)
        payload = {
            "row_id": row.get("row_id"),
            "run_id": row.get("run_id"),
            "status": "complete",
            "duration_seconds": time.perf_counter() - start,
            "path": str(run_dir),
            "device": str(device),
            "phase3": extra,
        }
        _write_status(status_path, payload)
        return payload
    except Exception as error:
        failure = {
            "row_id": row.get("row_id"),
            "run_id": row.get("run_id"),
            "status": "failed",
            "duration_seconds": time.perf_counter() - start,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "failure.json", failure)
        _write_status(status_path, failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one static Phase 3 manifest row.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--row-id", type=int, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    if args.row_id is not None:
        matches = [row for row in rows if int(row.get("row_id", -1)) == args.row_id]
        if not matches:
            raise SystemExit(f"No row_id={args.row_id} in {args.manifest}")
        row = matches[0]
    else:
        index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if index < 0 or index >= len(rows):
            raise SystemExit(f"array index {index} outside manifest of {len(rows)} rows")
        row = rows[index]
    result = execute_row(row, args.run_root, device_name=args.device, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
