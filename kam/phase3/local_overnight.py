from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kam.utils import choose_device

from .manifest import build_manifest, load_manifest
from .run_array import execute_row
from .table import read_json, write_json


_STOP = False


def _stop_handler(_signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def _estimate_p90(run_root: Path) -> float:
    durations: list[float] = []
    for path in (run_root / "status").glob("row_*.json"):
        try:
            payload = read_json(path)
            if payload.get("status") == "complete" and payload.get("duration_seconds") is not None:
                durations.append(float(payload["duration_seconds"]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return float(np.quantile(durations, 0.90)) if durations else 120.0


def detect_local_device() -> dict[str, Any]:
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return {"device": f"cuda:{index}", "gpu_name": properties.name, "vram_gib": properties.total_memory / (1024**3), "cuda": True}
    return {"device": "cpu", "gpu_name": None, "vram_gib": 0.0, "cuda": False}


def run_local(manifest: str | Path, run_root: str | Path, walltime_hours: float, resume: bool = True) -> dict[str, Any]:
    global _STOP
    _STOP = False
    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)
    manifest = Path(manifest)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    hardware = detect_local_device()
    device = choose_device("auto")
    rows = load_manifest(manifest)
    start = time.perf_counter()
    deadline = start + float(walltime_hours) * 3600.0
    completed = 0
    failed = 0
    skipped = 0
    for row in rows:
        if _STOP:
            break
        remaining = deadline - time.perf_counter()
        estimate = _estimate_p90(run_root)
        if remaining < max(60.0, estimate * 1.15):
            break
        result = execute_row(row, run_root, device_name=str(device), resume=resume)
        status = result.get("status")
        completed += status == "complete"
        skipped += status == "skipped_complete"
        failed += status == "failed"
        progress = {
            "hardware": hardware,
            "manifest": str(manifest),
            "run_root": str(run_root),
            "walltime_hours": walltime_hours,
            "elapsed_seconds": time.perf_counter() - start,
            "remaining_seconds": max(0.0, deadline - time.perf_counter()),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "last_result": result,
        }
        write_json(run_root / "local_progress.json", progress)
        print(json.dumps(progress, sort_keys=True, default=str), flush=True)
    summary = {
        "hardware": hardware,
        "manifest": str(manifest),
        "run_root": str(run_root),
        "elapsed_seconds": time.perf_counter() - start,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "stopped_by_signal": _STOP,
        "walltime_hours": walltime_hours,
    }
    write_json(run_root / "local_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the safe single-GPU Phase 3 overnight lane.")
    parser.add_argument("--manifest", type=Path, default=Path("results/phase3/manifests/local_overnight.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase3/local_overnight.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("results/phase3/local_overnight"))
    parser.add_argument("--walltime-hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.manifest.exists():
        build_manifest(args.config, args.manifest)
    hours = args.walltime_hours
    if hours is None:
        hours = float(os.environ.get("LOCAL_OVERNIGHT_HOURS", "10"))
    summary = run_local(args.manifest, args.run_root, hours, resume=args.resume)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
