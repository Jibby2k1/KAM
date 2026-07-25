"""Execute Stage 2 rows and aggregate independent held-out streams within seed."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
import torch
from kam.phase4.table import read_table
from kam.factory import make_model
from kam.run_suite import _build_data_with_test, _evaluate
from kam.phase5.pilot_run import execute_row
from kam.utils import save_json


def _heldout(metrics: dict[str, Any], row: dict[str, Any], run_root: Path, device: torch.device) -> list[dict[str, Any]]:
    checkpoint = Path(metrics["best_checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = make_model(payload["model_spec"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    rows = []
    count = int(row.get("heldout_streams", 1))
    for stream_index in range(count):
        stream_seed = int(row["seed"]) + 7000003 * (stream_index + 1)
        _train, _validation, test, task_type, metadata = _build_data_with_test(str(row["task"]), row, stream_seed)
        result = _evaluate(model, test, device, task_type, int(row.get("heldout_eval_batches", 512)))
        result.update({"stream_index": stream_index, "stream_seed": stream_seed, "task_generator": metadata.get("task_generator", metadata.get("symbolic_generator", False))})
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Stage 2 row.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_table(args.manifest)
    index = args.array_index if args.array_index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    row = rows[index]
    result = execute_row(row, args.run_root, args.device, args.resume)
    if result.get("status") not in {"complete", "skipped_complete"}:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1)
    run_dir = args.run_root / "runs" / str(row["run_id"])
    # A prior transient failure may have left a marker beside artifacts that
    # are now complete; the repaired run should clear that stale marker.
    (run_dir / "failure.json").unlink(missing_ok=True)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not (run_dir / "heldout_metrics.json").exists() or not args.resume:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        heldout_rows = _heldout(metrics, row, args.run_root, device)
        metrics["heldout_stream_metrics"] = heldout_rows
        total_n = sum(float(item.get("n", item.get("samples", 0.0))) for item in heldout_rows)
        if total_n:
            metrics["heldout_global_mse"] = sum(float(item.get("mse", 0.0)) * float(item.get("n", item.get("samples", 0.0))) for item in heldout_rows) / total_n
        save_json(run_dir / "heldout_metrics.json", {"streams": heldout_rows})
        save_json(metrics_path, metrics)
    try:
        print(json.dumps({"row_id": row["row_id"], "run_id": row["run_id"], "status": "complete", "heldout_streams": len(metrics.get("heldout_stream_metrics", []))}, indent=2, sort_keys=True), flush=True)
    except OSError:
        # A transient Lustre/stdout failure must not turn a completed row into
        # a false experiment failure after all artifacts were written.
        pass


if __name__ == "__main__":
    main()
