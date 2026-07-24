from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from kam.diagnostics import support_regime_metrics
from kam.factory import make_model
from kam.interventions import (
    ablate_branches,
    frozen_ridge_probe,
    perturb_memory,
    summarize_support,
    support_mask,
    support_rankings,
)
from kam.run_suite import _build_data, _move_batch, _masked_language_loss


def _inventory_row(checkpoint_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = checkpoint.get("model_spec", {})
    validation = checkpoint.get("validation", {})
    return {
        "checkpoint": str(checkpoint_path),
        "task": checkpoint.get("resolved_config", {}).get("run", {}).get("task", "unknown"),
        "model": spec.get("model_name", "unknown"),
        "parameters": sum(value.numel() for value in checkpoint.get("model_state", {}).values() if hasattr(value, "numel")),
        **{f"validation_{key}": value for key, value in validation.items()},
    }


def _loss_vector(model: nn.Module, batch: Any, device: torch.device, task_type: str) -> tuple[Tensor, Any, Tensor]:
    inputs, targets, mask = _move_batch(batch, device, task_type)
    with torch.no_grad():
        outputs, diagnostics = model(inputs, return_weights=True)
        if task_type == "regression":
            vector = (outputs - targets).square().reshape(-1)
        else:
            flat = nn.functional.cross_entropy(
                outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1), reduction="none"
            ).reshape_as(targets)
            vector = (flat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return vector.detach().cpu(), diagnostics, inputs


def _evaluate_loss(model: nn.Module, loader: Any, device: torch.device, task_type: str, max_batches: int) -> Tensor:
    losses = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        vector, _diagnostics, _inputs = _loss_vector(model, batch, device, task_type)
        losses.append(vector)
    if not losses:
        raise ValueError("Validation loader produced no batches.")
    return torch.cat(losses)


def _collect_baseline(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    task_type: str,
    max_batches: int,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, dict[str, float]]:
    losses: list[Tensor] = []
    weights: list[Tensor] = []
    features: list[Tensor] = []
    targets_for_probe: list[Tensor] = []
    regime_labels: list[Tensor] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        vector, diagnostics, inputs = _loss_vector(model, batch, device, task_type)
        losses.append(vector)
        if diagnostics.memory_weights:
            weights.append(diagnostics.memory_weights[-1].detach().cpu())
        if task_type == "regression":
            with torch.no_grad():
                encoded, _ = model.regression_features(inputs, return_weights=False)
            features.append(encoded.detach().cpu())
            targets_for_probe.append(_move_batch(batch, device, task_type)[1].detach().cpu().reshape(encoded.shape[0], -1))
        if isinstance(batch, dict) and "metadata" in batch:
            regime_labels.append(batch["metadata"].detach().cpu())
    if not losses:
        raise ValueError("Validation loader produced no batches.")
    all_weights = torch.cat(weights, dim=0) if weights else None
    all_features = torch.cat(features, dim=0) if features else None
    all_probe_targets = torch.cat(targets_for_probe, dim=0) if targets_for_probe else None
    all_regimes = torch.cat(regime_labels, dim=0) if regime_labels else None
    support_metrics = summarize_support(all_weights) if all_weights is not None else {}
    return torch.cat(losses), all_weights, all_features, all_probe_targets, {**support_metrics, "has_regime_metadata": float(all_regimes is not None)}


def _evaluate_run(checkpoint_path: Path, checkpoint: dict[str, Any], args: argparse.Namespace, diagnostics_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = checkpoint.get("model_spec")
    resolved_run = checkpoint.get("resolved_config", {}).get("run", {})
    metrics_path = checkpoint_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    task = str(resolved_run.get("task", metrics.get("task", "")))
    if not spec or not task:
        raise ValueError("Checkpoint does not contain enough metadata for reanalysis.")
    model = make_model(spec)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    task_type = str(spec["task_type"])
    _train_loader, validation_loader, _built_task_type, _data_metadata = _build_data(task, resolved_run, int(metrics.get("seed", resolved_run.get("seed", 7))))
    device = torch.device(args.device)
    model.to(device)
    losses, weights, features, probe_targets, support_metrics = _collect_baseline(model, validation_loader, device, task_type, args.max_batches)
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "task": task,
        "model": spec.get("model_name", "unknown"),
        "seed": metrics.get("seed", resolved_run.get("seed", 7)),
        "baseline_loss": float(losses.mean()),
        "context_ablation_loss": None,
        "memory_ablation_loss": None,
        "key_perturb_loss": None,
        "value_perturb_loss": None,
        **support_metrics,
    }
    with ablate_branches(model, context=True):
        row["context_ablation_loss"] = float(_evaluate_loss(model, validation_loader, device, task_type, args.max_batches).mean())
    with ablate_branches(model, memory=True):
        row["memory_ablation_loss"] = float(_evaluate_loss(model, validation_loader, device, task_type, args.max_batches).mean())
    with perturb_memory(model, key_noise=args.perturb_std, seed=19):
        row["key_perturb_loss"] = float(_evaluate_loss(model, validation_loader, device, task_type, args.max_batches).mean())
    with perturb_memory(model, value_noise=args.perturb_std, seed=23):
        row["value_perturb_loss"] = float(_evaluate_loss(model, validation_loader, device, task_type, args.max_batches).mean())
    row["context_ablation_delta"] = row["context_ablation_loss"] - row["baseline_loss"]
    row["memory_ablation_delta"] = row["memory_ablation_loss"] - row["baseline_loss"]
    row["key_perturb_delta"] = row["key_perturb_loss"] - row["baseline_loss"]
    row["value_perturb_delta"] = row["value_perturb_loss"] - row["baseline_loss"]
    if features is not None and probe_targets is not None:
        row.update(frozen_ridge_probe(features, probe_targets))
    curves: list[dict[str, Any]] = []
    if weights is not None:
        descending, ascending = support_rankings(weights)
        supports = weights.shape[-1]
        generator = torch.Generator().manual_seed(29)
        random_order = torch.randperm(supports, generator=generator)
        for requested in args.deletion_counts:
            count = min(int(requested), max(0, supports - 1))
            for kind, order in (("top", descending), ("random", random_order), ("bottom", ascending)):
                mask = torch.ones(supports, dtype=torch.bool)
                if count:
                    mask[order[:count]] = False
                with support_mask(model, mask):
                    deleted = _evaluate_loss(model, validation_loader, device, task_type, args.max_batches)
                curves.append({
                    "checkpoint": str(checkpoint_path),
                    "task": task,
                    "model": spec.get("model_name", "unknown"),
                    "deletion_count": count,
                    "deletion_kind": kind,
                    "loss": float(deleted.mean()),
                    "delta": float(deleted.mean() - losses.mean()),
                })
    raw_id = hashlib.sha1(str(checkpoint_path).encode()).hexdigest()[:16]
    raw_path = diagnostics_dir / f"{raw_id}.json"
    raw_payload = {
        "checkpoint": str(checkpoint_path),
        "summary": row,
        "baseline_losses": losses.tolist(),
        "support_mass": weights.mean(dim=(0, 1, 2)).tolist() if weights is not None else None,
        "deletion_curves": curves,
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    row["raw_diagnostics"] = str(raw_path)
    return row, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory or intervene on Phase II checkpoints.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/reanalysis_metrics.csv"))
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Run branch/perturbation/deletion diagnostics.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--perturb-std", type=float, default=0.1)
    parser.add_argument("--deletion-counts", type=int, nargs="+", default=[1, 4, 8])
    args = parser.parse_args()
    diagnostics_dir = args.diagnostics_dir or args.output.parent / "reanalysis_raw"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    for checkpoint_path in sorted(args.root.rglob("best_model.pt")):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if args.evaluate:
                row, curves = _evaluate_run(checkpoint_path, checkpoint, args, diagnostics_dir)
                rows.append(row)
                deletion_rows.extend(curves)
            else:
                rows.append(_inventory_row(checkpoint_path, checkpoint))
        except Exception as error:
            rows.append({"checkpoint": str(checkpoint_path), "status": "failed", "error": str(error)})
            if args.evaluate:
                failure_path = diagnostics_dir / (hashlib.sha1(str(checkpoint_path).encode()).hexdigest()[:16] + ".failure.json")
                failure_path.write_text(json.dumps({"checkpoint": str(checkpoint_path), "error": str(error), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if args.evaluate:
        deletion_output = args.output.with_name(args.output.stem + "_deletions.csv")
        fields = sorted({key for row in deletion_rows for key in row})
        with deletion_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(deletion_rows)
    print(f"Wrote {len(rows)} checkpoint rows to {args.output}")


if __name__ == "__main__":
    main()
