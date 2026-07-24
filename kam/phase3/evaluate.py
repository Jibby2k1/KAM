from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kam.data import (
    make_prototype_switch_splits,
    make_switching_mackey_splits,
    make_switching_narma_splits,
    schedule_segments,
)
from kam.diagnostics import support_utilization
from kam.factory import make_model
from kam.interventions import support_mask, support_rankings, summarize_support
from kam.online import nlms_update
from kam.run_suite import _build_data
from kam.utils import choose_device, save_json, set_seed


def _load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spec = checkpoint.get("model_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"Checkpoint does not contain model_spec: {path}")
    model = make_model(spec).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _make_splits(row: dict[str, Any], seed: int | None = None):
    task = str(row["task"])
    schedule = row.get("schedule", ["A", "B", "A"])
    if isinstance(schedule, str):
        schedule = schedule.replace("->", "-").split("-")
    length = int(row.get("online_length", row.get("series_length", 2400)))
    window = int(row.get("seq_len", 32))
    stream_seed = int(row.get("online_seed", seed if seed is not None else row.get("seed", 0) + 700_000))
    if task == "switching_mackey_glass":
        return make_switching_mackey_splits(total_length=length, window=window, schedule=schedule, seed=stream_seed)
    if task == "switching_narma":
        return make_switching_narma_splits(total_length=length, window=window, schedule=schedule, seed=stream_seed)
    if task == "prototype_switch":
        return make_prototype_switch_splits(length=length, window=window, schedule=schedule, seed=stream_seed)
    raise ValueError(f"Prequential Phase 3 evaluation does not support task={task}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prequential_evaluate(checkpoint_path: str | Path, row: dict[str, Any], output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(row.get("seed", 0)) + 900_000)
    static_model, _ = _load_checkpoint(checkpoint_path, device)
    adaptive_model = copy.deepcopy(static_model).to(device)
    adaptive_model.eval()
    for parameter in adaptive_model.parameters():
        parameter.requires_grad_(False)
    splits = _make_splits(row)
    values = (splits.raw_values.astype(np.float32) - float(splits.value_mean)) / max(float(splits.value_std), 1e-8)
    inputs = splits.raw_inputs.astype(np.float32)
    window = int(row.get("seq_len", 32))
    segments = schedule_segments(len(values), list(row.get("schedule", ["A", "B", "A"])))
    transitions = {segment.start for segment in segments[1:]}
    transition_window = int(row.get("recovery_window", min(256, max(16, len(values) // 20))))
    trace: list[dict[str, Any]] = []
    static_errors: list[float] = []
    adaptive_errors: list[float] = []
    for index in range(window, len(values)):
        features_np = np.stack([values[index - window : index], inputs[index - window : index]], axis=-1)
        model_input = torch.from_numpy(features_np).unsqueeze(0).to(device)
        target = torch.tensor([[values[index]]], dtype=torch.float32, device=device)
        with torch.no_grad():
            static_prediction = static_model(model_input)
            adaptive_features, _ = adaptive_model.regression_features(model_input, return_weights=False)
            adaptive_prediction = adaptive_model.readout(adaptive_features)
        static_error = float((static_prediction - target).squeeze().cpu())
        adaptive_error = float((adaptive_prediction - target).squeeze().cpu())
        # The update occurs only after both predictions have been scored.
        nlms_update(adaptive_model.readout, adaptive_features, target, eta=float(row.get("nlms_eta", 0.1)))
        phase = "pre_transition"
        transition_id = -1
        for transition_id_candidate, transition in enumerate(sorted(transitions)):
            if index >= transition:
                phase = "post_transition"
                transition_id = transition_id_candidate
        post_age = min((index - transition) for transition in transitions if index >= transition) if any(index >= transition for transition in transitions) else -1
        static_errors.append(static_error)
        adaptive_errors.append(adaptive_error)
        trace.append({
            "index": index,
            "phase": phase,
            "transition_id": transition_id,
            "post_transition_age": post_age,
            "target": float(target.squeeze().cpu()),
            "static_prediction": float(static_prediction.squeeze().cpu()),
            "adaptive_prediction": float(adaptive_prediction.squeeze().cpu()),
            "static_error": static_error,
            "adaptive_error": adaptive_error,
            "static_abs_error": abs(static_error),
            "adaptive_abs_error": abs(adaptive_error),
        })
    static_sq = np.square(np.asarray(static_errors, dtype=float))
    adaptive_sq = np.square(np.asarray(adaptive_errors, dtype=float))
    late_mask = np.asarray([row_item["post_transition_age"] >= 2 * transition_window for row_item in trace], dtype=bool)
    if not late_mask.any():
        late_mask = np.arange(len(trace)) >= max(0, len(trace) - transition_window)
    early_mask = np.asarray([0 <= row_item["post_transition_age"] < transition_window for row_item in trace], dtype=bool)
    if not early_mask.any():
        early_mask = np.arange(len(trace)) < min(transition_window, len(trace))
    static_late = float(static_sq[late_mask].mean())
    adaptive_late = float(adaptive_sq[late_mask].mean())
    static_early = float(static_sq[early_mask].mean())
    adaptive_early = float(adaptive_sq[early_mask].mean())
    metrics = {
        "run_id": row.get("run_id"),
        "task": row.get("task"),
        "variant": row.get("variant"),
        "seed": row.get("seed"),
        "online_seed": row.get("online_seed", int(row.get("seed", 0)) + 700_000),
        "n": len(trace),
        "static_mse": float(static_sq.mean()),
        "adaptive_mse": float(adaptive_sq.mean()),
        "static_early_post_transition_mse": static_early,
        "adaptive_early_post_transition_mse": adaptive_early,
        "static_late_post_transition_mse": static_late,
        "adaptive_late_post_transition_mse": adaptive_late,
        "relative_late_post_transition_nmse_improvement": float((static_late - adaptive_late) / max(abs(static_late), 1e-12)),
        "integrated_error_ratio": float(adaptive_sq.sum() / max(static_sq.sum(), 1e-12)),
        "transition_count": len(transitions),
        "transition_window": transition_window,
        "ordering": "predict_score_reveal_update",
    }
    _write_csv(output_dir / "prequential_trace.csv", trace)
    save_json(output_dir / "prequential_metrics.json", metrics)
    return metrics


def causal_probe(checkpoint_path: str | Path, row: dict[str, Any], output_dir: str | Path, device: torch.device) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = _load_checkpoint(Path(checkpoint_path), device)
    _, validation_loader, task_type, _ = _build_data(str(row["task"]), row, int(row.get("seed", 0)))
    if task_type != "regression" or not getattr(model, "use_memory", False):
        return {"run_id": row.get("run_id"), "status": "not_applicable"}
    batch = next(iter(validation_loader))
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        predictions, diagnostics = model(inputs, return_weights=True)
    if not diagnostics.memory_weights:
        return {"run_id": row.get("run_id"), "status": "not_applicable"}
    weights = diagnostics.memory_weights[-1]
    ranking, reverse_ranking = support_rankings(weights)
    support_summary = summarize_support(weights)
    baseline_losses = (predictions - targets).reshape(-1).square()
    supports = weights.shape[-1]
    requested = row.get("deletion_k", [1, 2, 4, 8, 16])
    if isinstance(requested, str):
        requested = json.loads(requested)
    counts = sorted({min(int(count), supports - 1) for count in requested if int(count) < supports})
    rng = np.random.default_rng(int(row.get("seed", 0)) + 4242)
    curves: list[dict[str, Any]] = []
    for count in counts:
        top_mask = torch.ones(supports, dtype=torch.bool, device=device)
        top_mask[ranking[:count]] = False
        bottom_mask = torch.ones(supports, dtype=torch.bool, device=device)
        bottom_mask[reverse_ranking[:count]] = False
        with torch.no_grad(), support_mask(model, top_mask):
            top_loss = float((model(inputs) - targets).reshape(-1).square().mean().cpu())
        with torch.no_grad(), support_mask(model, bottom_mask):
            bottom_loss = float((model(inputs) - targets).reshape(-1).square().mean().cpu())
        random_values: list[float] = []
        draws = int(row.get("deletion_draws", 8))
        for _ in range(draws):
            random_mask = torch.ones(supports, dtype=torch.bool, device=device)
            random_indices = rng.choice(supports, size=count, replace=False)
            random_mask[torch.as_tensor(random_indices, device=device)] = False
            with torch.no_grad(), support_mask(model, random_mask):
                random_values.append(float((model(inputs) - targets).reshape(-1).square().mean().cpu()))
        base = float(baseline_losses.mean().cpu())
        curves.append({
            "run_id": row.get("run_id"),
            "task": row.get("task"),
            "variant": row.get("variant"),
            "scale": row.get("scale"),
            "seed": row.get("seed"),
            "deletion_count": count,
            "baseline_loss": base,
            "top_loss": top_loss,
            "bottom_loss": bottom_loss,
            "random_loss_mean": float(np.mean(random_values)),
            "random_loss_std": float(np.std(random_values)),
            "top_delta": top_loss - base,
            "random_delta": float(np.mean(random_values)) - base,
            "bottom_delta": bottom_loss - base,
        })
    _write_csv(output_dir / "deletion_curves.csv", curves)
    result = {"run_id": row.get("run_id"), "status": "complete", **support_summary, "deletion_rows": len(curves)}
    save_json(output_dir / "support_diagnostics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase 3 prequential or causal probe for one checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--series-length", type=int, default=2400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = {"task": args.task, "variant": args.variant, "seq_len": args.seq_len, "series_length": args.series_length, "online_length": args.series_length, "seed": 0, "run_id": args.checkpoint.parent.name}
    device = choose_device(args.device)
    print(json.dumps(prequential_evaluate(args.checkpoint, row, args.output, device), indent=2))


if __name__ == "__main__":
    main()
