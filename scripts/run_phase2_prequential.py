from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kam.adaptation import make_adapter, prequential_regression
from kam.factory import make_model
from kam.run_suite import _build_data


def _load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = checkpoint["model_spec"]
    model = make_model(spec)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    resolved = checkpoint.get("resolved_config", {}).get("run", {})
    metrics_path = path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    task = str(resolved.get("task", metrics.get("task", "unknown")))
    seed = int(resolved.get("seed", metrics.get("seed", 7)))
    return checkpoint, model, resolved, task, seed


def run_one(path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    device = torch.device(args.device)
    checkpoint, model, resolved, task, seed = _load_checkpoint(path, device)
    if checkpoint["model_spec"].get("task_type") != "regression":
        raise ValueError(f"Prequential regression requires a regression checkpoint: {path}")
    _train, validation, task_type, _metadata = _build_data(task, resolved, seed)
    loader = DataLoader(validation.dataset, batch_size=1, shuffle=False)
    first_batch = next(iter(loader))
    with torch.no_grad():
        first_features, _ = model.regression_features(first_batch[0].to(device), return_weights=False)
    feature_dim = first_features.shape[-1]
    rows: list[dict[str, object]] = []
    for adapter_name in args.adapters:
        adapter_kwargs = {"eta": args.eta} if adapter_name in {"nlms", "sgd"} else ({"forgetting": args.forgetting} if adapter_name == "rls" else {})
        adapter = make_adapter(adapter_name, feature_dim, device=device, **adapter_kwargs)
        with torch.no_grad():
            adapter.linear.weight.copy_(model.readout.weight)
            adapter.linear.bias.copy_(model.readout.bias)
        def feature_fn(inputs: torch.Tensor) -> torch.Tensor:
            return model.regression_features(inputs, return_weights=False)[0]
        stream = ((inputs, targets) for inputs, targets in itertools.islice(loader, args.max_samples))
        start = time.perf_counter()
        result = prequential_regression(feature_fn, adapter, stream, device=device)
        rows.append({
            "checkpoint": str(path),
            "task": task,
            "variant": checkpoint["model_spec"].get("model_name", "unknown"),
            "seed": seed,
            "adapter": adapter_name,
            "samples": int(result.metrics["samples"]),
            "mse": result.metrics["mse"],
            "mae": result.metrics["mae"],
            "seconds": time.perf_counter() - start,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predict-then-score-then-reveal prequential adapters.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/prequential_metrics.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--adapters", nargs="+", default=["frozen", "nlms", "sgd", "rls"])
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--forgetting", type=float, default=1.0)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for checkpoint in args.checkpoint:
        try:
            rows.extend(run_one(checkpoint, args))
        except Exception as error:
            failures.append({"checkpoint": str(checkpoint), "error": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        args.output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    else:
        stale_failures = args.output.with_suffix(".failures.json")
        if stale_failures.exists():
            stale_failures.unlink()
    print(f"Wrote {len(rows)} prequential rows to {args.output}; failures={len(failures)}")


if __name__ == "__main__":
    main()
