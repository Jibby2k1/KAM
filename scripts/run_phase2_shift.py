from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kam.adaptation import make_adapter, prequential_regression
from kam.diagnostics import support_regime_metrics, support_utilization
from kam.data import make_switching_mackey_splits, make_switching_narma_splits, SwitchingRegressionDataset
from kam.factory import make_model


def _schedule(value):
    if isinstance(value, str):
        return value.replace("->", "-").split("-")
    return list(value or ["A", "B", "A"])


def _load(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["model_spec"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    resolved = checkpoint.get("resolved_config", {}).get("run", {})
    task = str(resolved.get("task", checkpoint.get("task", "")))
    seed = int(resolved.get("seed", checkpoint.get("seed", 7)))
    return checkpoint, model, resolved, task, seed


def _transition_metrics(losses: np.ndarray, regimes: np.ndarray, *, horizon: int) -> list[dict[str, object]]:
    transitions = np.flatnonzero(regimes[1:] != regimes[:-1]) + 1
    rows: list[dict[str, object]] = []
    for transition in transitions:
        stop = min(len(losses), int(transition + horizon))
        early = losses[int(transition) : stop]
        next_transition = np.flatnonzero(transitions > transition)
        segment_stop = int(transitions[next_transition[0]]) if len(next_transition) else len(losses)
        late_start = max(int(transition), segment_stop - horizon)
        late = losses[late_start:segment_stop]
        steady = float(late.mean()) if len(late) else float(early.mean())
        threshold = steady * 1.10 + 1e-12
        recovery = None
        rolling = max(4, min(horizon // 4, len(losses) - int(transition)))
        for index in range(int(transition), max(int(transition), segment_stop - rolling + 1)):
            if float(losses[index : index + rolling].mean()) <= threshold:
                recovery = index - int(transition)
                break
        rows.append({
            "transition_index": int(transition),
            "from_regime": str(regimes[transition - 1]),
            "to_regime": str(regimes[transition]),
            "early_loss": float(early.mean()) if len(early) else float("nan"),
            "late_loss": float(late.mean()) if len(late) else float("nan"),
            "recovery_steps": recovery,
            "forgetting_ratio": float(late.mean() / max(float(early.mean()), 1e-12)) if len(early) and len(late) else float("nan"),
        })
    return rows


def run_checkpoint(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device(args.device)
    checkpoint, model, resolved, task, seed = _load(path, device)
    window = int(checkpoint["model_spec"]["max_seq_len"])
    requested_schedules = args.schedules or [resolved.get("schedule", ["A", "B", "A"])]
    total_length = int(args.series_length) if int(args.series_length) > 0 else int(resolved.get("series_length", 6000))
    regimes = resolved.get("regimes")
    trace_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for schedule_index, requested_schedule in enumerate(requested_schedules):
        schedule = _schedule(requested_schedule)
        stream_seed = seed + int(args.stream_seed_stride) * schedule_index
        if task == "switching_mackey_glass":
            splits = make_switching_mackey_splits(total_length=total_length, window=window, schedule=schedule, regimes=regimes, seed=stream_seed)
        elif task == "switching_narma":
            splits = make_switching_narma_splits(total_length=total_length, window=window, schedule=schedule, regimes=regimes, seed=stream_seed)
        else:
            raise ValueError(f"not a switching checkpoint: {task}")
        full = SwitchingRegressionDataset(splits.raw_values, splits.raw_inputs, splits.labels, window=window, value_mean=splits.value_mean, value_std=splits.value_std)
        loader = DataLoader(full, batch_size=args.feature_batch_size, shuffle=False)
        regimes_for_targets = full.target_regimes()
        _regime_names, regime_codes = np.unique(regimes_for_targets, return_inverse=True)
        feature_batches = []
        target_batches = []
        weight_batches = []
        assignment_batches = []
        with torch.no_grad():
            for inputs, targets in loader:
                features, diagnostics = model.regression_features(inputs.to(device), return_weights=True)
                feature_batches.append(features.detach())
                target_batches.append(targets.to(device))
                if diagnostics.memory_weights:
                    weights = diagnostics.memory_weights[-1].detach().cpu()
                    weight_batches.append(weights)
                    assignment_batches.append(weights[:, :, -1, :].mean(dim=1).argmax(dim=-1))
        all_features = torch.cat(feature_batches, dim=0)
        all_targets = torch.cat(target_batches, dim=0)
        feature_dim = int(all_features.shape[-1])
        for adapter_name in args.adapters:
            kwargs = {"eta": args.eta} if adapter_name in {"nlms", "sgd"} else ({"forgetting": args.forgetting} if adapter_name == "rls" else {})
            adapter = make_adapter(adapter_name, feature_dim, device=device, **kwargs)
            with torch.no_grad():
                adapter.linear.weight.copy_(model.readout.weight)
            losses = []
            predictions = []
            targets = []
            for index in range(all_features.shape[0]):
                features = all_features[index : index + 1]
                target = all_targets[index : index + 1]
                with torch.no_grad():
                    prediction = adapter.predict(features)
                    loss = (prediction - target).square()
                predictions.append(prediction.detach().cpu())
                targets.append(target.detach().cpu())
                losses.append(loss.detach().cpu())
                adapter.update(features, target)
            losses = torch.cat(losses).reshape(-1).numpy()
            predictions = torch.cat(predictions).reshape(-1).numpy()
            targets = torch.cat(targets).reshape(-1).numpy()
            transition_rows = _transition_metrics(losses, regimes_for_targets, horizon=args.horizon)
            support_summary = {}
            if weight_batches:
                all_weights = torch.cat(weight_batches, dim=0)
                support_summary = support_utilization(all_weights)
                support_summary.update({f"support_{key}": value for key, value in support_regime_metrics(torch.cat(assignment_batches), torch.from_numpy(regime_codes.astype(np.int64))).items()})
            for transition in transition_rows:
                transition.update({"checkpoint": str(path), "task": task, "variant": checkpoint["model_spec"].get("model_name", "unknown"), "adapter": adapter_name, "seed": seed, "schedule": "-".join(schedule), "schedule_index": schedule_index, "stream_seed": stream_seed, **support_summary})
                summary_rows.append(transition)
            for index, (loss, prediction, target, regime) in enumerate(zip(losses, predictions, targets, regimes_for_targets)):
                trace_rows.append({"checkpoint": str(path), "task": task, "variant": checkpoint["model_spec"].get("model_name", "unknown"), "adapter": adapter_name, "seed": seed, "schedule": "-".join(schedule), "schedule_index": schedule_index, "stream_seed": stream_seed, "index": index, "regime": str(regime), "loss": float(loss), "prediction": float(prediction), "target": float(target)})
    return trace_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prequential adaptation and recovery on switching streams.")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[], help="Recursively evaluate best_model.pt files below each root.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2/switching_adaptation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--series-length", type=int, default=0, help="Override stream length; 0 uses the checkpoint training stream length.")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--adapters", nargs="+", default=["frozen", "nlms", "sgd", "rls"])
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--forgetting", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--no-trace", action="store_true", help="Write transition summaries without retaining per-sample prediction traces.")
    parser.add_argument("--schedule", dest="schedules", action="append", default=None, help="Held-out schedule such as A-B-A-C-A; repeat for multiple schedules.")
    parser.add_argument("--stream-seed-stride", type=int, default=100003)
    args = parser.parse_args()
    checkpoints = list(args.checkpoint)
    for root in args.checkpoint_root:
        checkpoints.extend(sorted(root.rglob("best_model.pt")))
    if not checkpoints:
        parser.error("at least one --checkpoint or --checkpoint-root is required")
    trace_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    failures = []
    for checkpoint in checkpoints:
        try:
            trace, summary = run_checkpoint(checkpoint, args)
            trace_rows.extend(trace)
            summary_rows.extend(summary)
        except Exception as error:
            failures.append({"checkpoint": str(checkpoint), "error": str(error)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = [("shift_metrics.csv", summary_rows)]
    if not args.no_trace:
        output_rows.insert(0, ("shift_trace.csv", trace_rows))
    for name, rows in output_rows:
        fields = sorted({key for row in rows for key in row})
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    if failures:
        (args.output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    elif (args.output_dir / "failures.json").exists():
        (args.output_dir / "failures.json").unlink()
    print(f"Wrote {len(trace_rows)} trace rows and {len(summary_rows)} transition rows; failures={len(failures)}")


if __name__ == "__main__":
    main()
