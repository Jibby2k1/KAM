from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import (
    BoundedDyck2Dataset,
    CopyLanguageDataset,
    VariableCopyLanguageDataset,
    variable_copy_collate,
    MQARDataset,
    RegimeGrammarDataset,
    make_mackey_splits,
    make_narma_splits,
    make_prototype_switch_splits,
    make_switching_mackey_splits,
    make_switching_narma_splits,
)
from .factory import make_model
from .experiment_registry import ExperimentRegistry
from .capacity import capacity_summary
from .data.controlled_regimes import ControlledWindowDataset, make_independent_controlled_streams
from .utils import atomic_torch_save, choose_device, json_ready, save_json, set_seed
from .phase3.memory_trace import (
    memory_bank_parameters,
    set_memory_bank_trainable,
    snapshot_memory_bank,
    trace_row,
)


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def _masked_language_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, float]]:
    flat = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    mask = mask.to(dtype=flat.dtype)
    denominator = mask.sum().clamp_min(1.0)
    loss = (flat * mask).sum() / denominator
    accuracy = (((logits.argmax(dim=-1) == targets).to(mask.dtype) * mask).sum() / denominator)
    return loss, {
        "cross_entropy": float(loss.detach()),
        "bits_per_token": float((loss.detach() / np.log(2.0))),
        "accuracy": float(accuracy.detach()),
        "perplexity": float(torch.exp(loss.detach().clamp(max=20.0))),
    }


def _descriptive_regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = predictions.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    errors = predictions - targets
    absolute = errors.abs()
    mse = errors.square().mean()
    target_centered = targets - targets.mean()
    denominator = target_centered.square().sum().clamp_min(1e-12)
    prediction_centered = predictions - predictions.mean()
    correlation_denominator = prediction_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
    correlation = (prediction_centered * target_centered).sum() / correlation_denominator.clamp_min(1e-12)
    target_scale = targets.abs().mean().clamp_min(1e-12)
    median_abs = absolute.median()
    p90_abs = torch.quantile(absolute, 0.90)
    p95_abs = torch.quantile(absolute, 0.95)
    return {
        "n": float(predictions.numel()),
        "mse": float(mse),
        "rmse": float(mse.sqrt()),
        "mae": float(absolute.mean()),
        "bias": float(errors.mean()),
        "error_std": float(errors.std(unbiased=False)),
        "median_abs_error": float(median_abs),
        "p90_abs_error": float(p90_abs),
        "p95_abs_error": float(p95_abs),
        "max_abs_error": float(absolute.max()),
        "r2": float(1.0 - errors.square().sum() / denominator),
        "correlation": float(correlation),
        "relative_mae": float(absolute.mean() / target_scale),
        "nmse": float(mse / target_centered.square().mean().clamp_min(1e-12)),
        "nrmse": float((mse / target_centered.square().mean().clamp_min(1e-12)).sqrt()),
        "target_variance": float(target_centered.square().mean()),
        "p95_to_median_abs_error": float(p95_abs / median_abs.clamp_min(1e-12)),
        "log10_median_abs_error": float(torch.log10(median_abs.clamp_min(1e-12))),
    }


@torch.no_grad()
def _collect_regression_predictions(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> tuple[Tensor, Tensor, list[dict[str, float]]]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    rows: list[dict[str, float]] = []
    offset = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs, batch_targets = batch
        outputs = model(inputs.to(device)).detach().float().reshape(-1)
        batch_targets = batch_targets.to(device).detach().float().reshape(-1)
        predictions.append(outputs.cpu())
        targets.append(batch_targets.cpu())
        errors = outputs - batch_targets
        for local_index, (prediction, target, error) in enumerate(zip(outputs, batch_targets, errors)):
            rows.append({
                "index": float(offset + local_index),
                "target": float(target),
                "prediction": float(prediction),
                "error": float(error),
                "abs_error": float(error.abs()),
            })
        offset += int(outputs.numel())
    if not predictions:
        raise ValueError("Validation prediction collection produced no samples.")
    return torch.cat(predictions), torch.cat(targets), rows


PHASE5_CONTROLLED_TASKS = {
    "controlled_prototype",
    "controlled_symbolic_regime_language",
    "switching_mackey_glass_controlled",
    "switching_narma_controlled",
}

REGRESSION_TASKS = PHASE5_CONTROLLED_TASKS | {
    "switching_mackey_glass",
    "switching_narma",
    "prototype_switch",
    "mackey_glass",
    "narma",
}


def _build_regression_data(
    task: str, params: dict[str, Any], seed: int
) -> tuple[DataLoader, DataLoader, DataLoader, str, dict[str, Any]]:
    """Build train/validation/test loaders for continuous regression streams."""

    batch_size = int(params.get("batch_size", 32))
    if task in PHASE5_CONTROLLED_TASKS:
        stream_kwargs = {
            "regime_count": int(params.get("regime_count", 2)),
            "regime_separation": params.get("regime_separation", "medium"),
            "return_probability": float(params.get("return_probability", 0.5)),
            "dwell_length": int(params.get("dwell_length", max(8, int(params.get("seq_len", 32)) * 2))),
            "transition_type": str(params.get("transition_type", "abrupt")),
            "observation_noise": float(params.get("observation_noise", 0.0)),
            "process_noise": float(params.get("process_noise", 0.0)),
            "input_noise": float(params.get("input_noise", 0.0)),
            "observability": str(params.get("observability", "full")),
        }
        lengths = {
            "train": int(params.get("train_length", params.get("series_length", 1200))),
            "validation": int(params.get("validation_length", max(256, int(params.get("series_length", 1200) // 3)))),
            "test": int(params.get("test_length", max(256, int(params.get("series_length", 1200) // 3)))),
            "prequential": int(params.get("prequential_length", max(256, int(params.get("series_length", 1200) // 3)))),
        }
        streams = make_independent_controlled_streams(lengths=lengths, seed=seed, **stream_kwargs)
        window = int(params.get("seq_len", 32))
        train = ControlledWindowDataset(streams["train"], window)
        validation = ControlledWindowDataset(streams["validation"], window)
        test = ControlledWindowDataset(streams["test"], window)
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window,
            "controlled_streams": True, "independent_split_streams": True,
            "stream_metadata": {name: stream.metadata for name, stream in streams.items()},
            **stream_kwargs,
        }
        return (
            DataLoader(train, batch_size=batch_size, shuffle=bool(params.get("training_protocol", "iid_window_training") == "iid_window_training")),
            DataLoader(validation, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size),
            "regression",
            metadata,
        )
    window = int(params.get("seq_len", 32))
    schedule = params.get("schedule", ["A", "B", "A"])
    if isinstance(schedule, str):
        schedule = schedule.replace("->", "-").split("-")
    total_length = int(params.get("series_length", 6000))
    if task == "switching_mackey_glass":
        splits = make_switching_mackey_splits(
            total_length=total_length,
            window=window,
            schedule=schedule,
            regimes=params.get("regimes"),
            seed=seed,
        )
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window,
            "schedule": list(schedule), "regimes": params.get("regimes"), "switching": True,
        }
    elif task == "switching_narma":
        splits = make_switching_narma_splits(
            total_length=total_length,
            window=window,
            schedule=schedule,
            regimes=params.get("regimes"),
            seed=seed,
        )
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window,
            "schedule": list(schedule), "regimes": params.get("regimes"), "switching": True,
        }
    elif task == "prototype_switch":
        splits = make_prototype_switch_splits(
            length=total_length, window=window, schedule=schedule, seed=seed
        )
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window,
            "schedule": list(schedule), "regimes": params.get("regimes"), "switching": True,
        }
    elif task == "mackey_glass":
        tau = float(params.get("tau", 17.0))
        beta = float(params.get("beta", 0.2))
        splits = make_mackey_splits(
            total_length=total_length, window=window, tau=tau, beta=beta, seed=seed
        )
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window,
            "tau": tau, "beta": beta,
        }
    elif task == "narma":
        order = int(params.get("order", 10))
        splits = make_narma_splits(
            length=total_length, order=order, window=window, seed=seed
        )
        metadata = {
            "input_dim": 2, "output_dim": 1, "max_seq_len": window, "order": order,
        }
    else:
        raise ValueError(f"Unsupported regression task: {task}")
    return (
        DataLoader(splits.train, batch_size=batch_size, shuffle=True),
        DataLoader(splits.validation, batch_size=batch_size),
        DataLoader(splits.test, batch_size=batch_size),
        "regression",
        metadata,
    )


def _build_data(task: str, params: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader, str, dict[str, Any]]:
    batch_size = int(params.get("batch_size", 32))
    train_size = int(params.get("train_size", 512))
    val_size = int(params.get("val_size", 128))
    if task == "mqar":
        common = {
            "sequence_length": int(params.get("sequence_length", 64)),
            "num_bindings": int(params.get("bindings", 8)),
            "num_queries": int(params.get("queries", 4)),
            "vocab_size": int(params.get("vocab_size", 128)),
        }
        train = MQARDataset(size=train_size, seed=seed, **common)
        validation = MQARDataset(size=val_size, seed=seed + 1_000_000, **common)
        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(validation, batch_size=batch_size), "language", common | {"vocab_size": train.vocab_size}
    if task == "copy":
        length = int(params.get("copy_length", 16))
        common = {"payload_length": length, "alphabet_size": int(params.get("alphabet_size", 17))}
        train = CopyLanguageDataset(size=train_size, seed=seed, **common)
        validation = CopyLanguageDataset(size=val_size, seed=seed + 1_000_000, **common)
        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(validation, batch_size=batch_size), "language", {"vocab_size": train.vocab_size, "max_seq_len": train.sequence_length, **common}
    if task == "variable_copy":
        common = {
            "min_payload_length": int(params.get("min_copy_length", 4)),
            "max_payload_length": int(params.get("max_copy_length", 32)),
            "alphabet_size": int(params.get("alphabet_size", 17)),
        }
        train = VariableCopyLanguageDataset(size=train_size, seed=seed, **common)
        validation = VariableCopyLanguageDataset(size=val_size, seed=seed + 1_000_000, **common)
        collate = variable_copy_collate
        return (DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate),
                DataLoader(validation, batch_size=batch_size, collate_fn=collate),
                "language", {"vocab_size": train.vocab_size, "max_seq_len": train.max_sequence_length, **common})

    if task == "dyck2":
        max_depth = int(params.get("max_depth", 8))
        train = BoundedDyck2Dataset(size=train_size, max_depth=max_depth, seed=seed)
        validation = BoundedDyck2Dataset(size=val_size, max_depth=max_depth, seed=seed + 1_000_000)
        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(validation, batch_size=batch_size), "language", {"vocab_size": train.vocab_size, "max_seq_len": train.sequence_length, "max_depth": max_depth}
    if task == "regime":
        seq_len = int(params.get("seq_len", 64))
        train = RegimeGrammarDataset(size=train_size, sequence_length=seq_len, seed=seed)
        validation = RegimeGrammarDataset(size=val_size, sequence_length=seq_len, seed=seed + 1_000_000, switch_halfway=bool(params.get("switch_validation", False)))
        return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(validation, batch_size=batch_size), "language", {"vocab_size": train.vocab_size, "max_seq_len": seq_len + 1, "switch_validation": bool(params.get("switch_validation", False))}
    if task in REGRESSION_TASKS:
        train_loader, validation_loader, _test_loader, task_type, data_meta = _build_regression_data(task, params, seed)
        return train_loader, validation_loader, task_type, data_meta
    raise ValueError(f"Unsupported Phase II task: {task}")


def _build_data_with_test(
    task: str, params: dict[str, Any], seed: int
) -> tuple[DataLoader, DataLoader, DataLoader | None, str, dict[str, Any]]:
    """Return an optional held-out test loader without changing legacy callers."""

    if task in REGRESSION_TASKS:
        return _build_regression_data(task, params, seed)
    train_loader, validation_loader, task_type, data_meta = _build_data(task, params, seed)
    return train_loader, validation_loader, None, task_type, data_meta


def _move_batch(batch: Any, device: torch.device, task_type: str) -> tuple[Tensor, Tensor, Tensor | None]:
    if task_type == "regression":
        inputs, targets = batch
        return inputs.to(device), targets.to(device), None
    return batch["inputs"].to(device), batch["targets"].to(device), batch["loss_mask"].to(device)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, task_type: str, max_batches: int = 50) -> dict[str, float]:
    model.eval()
    if task_type == "regression":
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            inputs, batch_targets, _ = _move_batch(batch, device, task_type)
            predictions.append(model(inputs).detach().float().reshape(-1).cpu())
            targets.append(batch_targets.detach().float().reshape(-1).cpu())
        if not predictions:
            return {}
        return _descriptive_regression_metrics(torch.cat(predictions), torch.cat(targets))
    totals: dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        inputs, targets, mask = _move_batch(batch, device, task_type)
        _, metrics = _masked_language_loss(model(inputs), targets, mask)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def _initialize_data_centers(model: nn.Module, loader: DataLoader) -> None:
    """Initialize KC-LV keys from observed training representations, then freeze them."""
    batch = next(iter(loader))
    inputs = batch[0]
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    hidden = model.input_layer(inputs)
    hidden = hidden + model._position_encoding(hidden.shape[1], device, hidden.dtype)
    for block in getattr(model, "blocks", []):
        memory = getattr(block, "memory", None)
        if memory is None:
            continue
        head_dim = int(memory.head_dim)
        reshaped = hidden.reshape(hidden.shape[0] * hidden.shape[1], memory.num_heads, head_dim)
        indices = torch.linspace(0, reshaped.shape[0] - 1, memory.num_supports, device=device).long()
        centers = reshaped[indices].permute(1, 0, 2).contiguous()
        memory.memory_keys.copy_(centers)
        memory.memory_keys.requires_grad_(False)


def _configure_phase5_variant(model: nn.Module, variant: str, train_loader: DataLoader) -> dict[str, Any]:
    metadata: dict[str, Any] = {"variant_semantics": variant}
    if variant == "RK-LV":
        for name, parameter in model.named_parameters():
            if name.endswith("memory_keys"):
                parameter.requires_grad_(False)
        metadata["fixed_component"] = "random_keys"
    elif variant == "LK-RV":
        for name, parameter in model.named_parameters():
            if name.endswith("memory_values"):
                parameter.requires_grad_(False)
        metadata["fixed_component"] = "random_values"
    elif variant == "KC-LV":
        _initialize_data_centers(model, train_loader)
        metadata["fixed_component"] = "sampled_training_representation_keys"
    elif variant == "RFF":
        metadata["fixed_component"] = "random_fourier_map"
    return metadata


def _evaluation_loader(loader: DataLoader) -> DataLoader:
    """Create a deterministic, non-shuffled view of an existing loader."""

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size or 32,
        shuffle=False,
        collate_fn=loader.collate_fn,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_one(run: dict[str, Any], root: Path, device: torch.device, precision: str) -> dict[str, Any]:
    seed = int(run.get("seed", 7))
    set_seed(seed)
    task = str(run["task"])
    train_loader, validation_loader, test_loader, task_type, data_meta = _build_data_with_test(task, run, seed)
    spec = {
        "model_name": run.get("variant", run.get("model_name", "D0")),
        "task_type": task_type,
        "d_model": int(run.get("d_model", 48)),
        "num_heads": int(run.get("num_heads", 4)),
        "num_layers": int(run.get("num_layers", 1)),
        "num_supports": int(run.get("num_supports", 32)),
        "max_seq_len": int(data_meta.get("max_seq_len", run.get("sequence_length", run.get("seq_len", 64))) + int(run.get("max_seq_len_extra", 0))),
        **data_meta,
        "context_window": run.get("context_window"),
        "dropout": float(run.get("dropout", 0.0)),
        "memory_mode": run.get("memory_mode", "both"),
        "memory_output": run.get("memory_output", run.get("memory_mode", "both")),
        "route_features": run.get("route_features", "raw"),
        "route_projection_dim": run.get("route_projection_dim"),
        "radial_metric": run.get("radial_metric", "diagonal"),
        "bandwidth": run.get("bandwidth", "learned"),
        "bandwidth_init": float(run.get("bandwidth_init", 1.0)),
        "ffn_expansion": int(run.get("ffn_expansion", 4)),
        "expose_memory_weights": bool(run.get("expose_memory_weights", False)),
        "parameter_match_target": run.get("parameter_match_target"),
        "position_mode": run.get("position_mode", "learned"),
        "fourier_features": run.get("fourier_features"),
    }
    model = make_model(spec).to(device)
    variant_metadata = _configure_phase5_variant(model, str(run.get("variant", spec["model_name"])), train_loader)
    capacity = capacity_summary(model, spec, int(spec["max_seq_len"]))
    steps = int(run.get("steps", 10))
    eval_every = int(run.get("eval_every", max(1, steps // 10)))
    memory_protocol = str(run.get("memory_protocol", "joint"))
    if spec["model_name"].endswith("-staged") and "memory_protocol" not in run:
        memory_protocol = "warmup_then_freeze"
    if memory_protocol not in {"joint", "warmup_then_freeze"}:
        raise ValueError(f"Unsupported memory_protocol={memory_protocol!r}")
    memory_parameters = memory_bank_parameters(model)
    if memory_protocol == "warmup_then_freeze":
        if not memory_parameters:
            raise ValueError("warmup_then_freeze requires a learned memory bank")
        if spec["model_name"] in {"RF-b", "RF-b-readout"}:
            raise ValueError("Random frozen controls cannot use warmup_then_freeze")
        if steps < 2:
            raise ValueError("warmup_then_freeze requires at least two training steps")
        warmup_fraction = float(run.get("memory_warmup_fraction", 0.75))
        if not 0.0 < warmup_fraction < 1.0:
            raise ValueError("memory_warmup_fraction must lie strictly between 0 and 1")
        freeze_step = int(run.get("memory_freeze_step", round(steps * warmup_fraction)))
        freeze_step = min(max(1, freeze_step), steps - 1)
        set_memory_bank_trainable(model, True)
    else:
        warmup_fraction = None
        freeze_step = None
    initial_memory_trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    memory_bank_parameter_count = sum(parameter.numel() for parameter in memory_parameters.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(run.get("learning_rate", 3e-4)), weight_decay=float(run.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    autocast_enabled = precision in {"amp", "bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if precision in {"amp", "bf16"} else torch.float16
    output_dir = root / str(run["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {"run": run, "model_spec": spec, "data_metadata": data_meta, "precision": precision, **_git_metadata()}
    save_json(output_dir / "resolved_config.json", resolved)
    best_metric = float("inf")
    best_validation: dict[str, float] | None = None
    best_step = 0
    history: list[dict[str, Any]] = []
    memory_trace_enabled = bool(run.get("memory_trace", False)) or memory_protocol == "warmup_then_freeze"
    evaluate_train = bool(run.get("evaluate_train", False)) or memory_protocol == "warmup_then_freeze"
    evaluate_test = bool(run.get("evaluate_test", False)) or memory_protocol == "warmup_then_freeze"
    trace_test = bool(run.get("trace_test", True)) and test_loader is not None
    trace_eval_every = max(1, int(run.get("trace_eval_every", eval_every)))
    train_eval_loader = _evaluation_loader(train_loader) if evaluate_train or memory_trace_enabled else None
    probe_inputs: Tensor | None = None
    if memory_trace_enabled and memory_parameters and train_eval_loader is not None:
        probe_batch = next(iter(train_eval_loader))
        probe_inputs, _probe_targets, _probe_mask = _move_batch(probe_batch, device, task_type)
    initial_memory = snapshot_memory_bank(model)
    previous_memory: dict[str, Tensor] | None = None
    memory_trace_rows: list[dict[str, Any]] = []
    memory_support_rows: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if memory_protocol == "warmup_then_freeze" and step == int(freeze_step) + 1:
            set_memory_bank_trainable(model, False)
        if memory_protocol == "warmup_then_freeze":
            stage = "memory_adaptation" if step <= int(freeze_step) else "backbone_finetuning"
        elif memory_parameters and not any(parameter.requires_grad for parameter in memory_parameters.values()):
            stage = "memory_frozen_from_init"
        else:
            stage = "joint_training"
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs, targets, mask = _move_batch(batch, device, task_type)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(inputs)
            if task_type == "regression":
                loss = nn.functional.mse_loss(outputs, targets)
                train_metrics = {"mse": float(loss.detach()), "mae": float(nn.functional.l1_loss(outputs, targets).detach())}
            else:
                loss, train_metrics = _masked_language_loss(outputs, targets, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(run.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(run.get("grad_clip", 1.0)))
            optimizer.step()
        trace_due = memory_trace_enabled and (step == 1 or step % trace_eval_every == 0 or step == steps or step in {freeze_step, (freeze_step + 1 if freeze_step is not None else -1)})
        should_evaluate = step == 1 or step % eval_every == 0 or step == steps or trace_due
        if should_evaluate:
            validation = _evaluate(model, validation_loader, device, task_type, int(run.get("eval_batches", 20)))
            primary = validation["mse"] if task_type == "regression" else validation["cross_entropy"]
            record: dict[str, Any] = {"step": step, "train": train_metrics, "validation": validation, "elapsed_seconds": time.perf_counter() - start, "stage": stage}
            if trace_due:
                train_evaluation = _evaluate(model, train_eval_loader, device=device, task_type=task_type, max_batches=int(run.get("trace_eval_batches", run.get("eval_batches", 20)))) if train_eval_loader is not None else {}
                test_evaluation = _evaluate(model, test_loader, device=device, task_type=task_type, max_batches=int(run.get("trace_eval_batches", run.get("eval_batches", 20)))) if trace_test and test_loader is not None else {}
                memory_summary: dict[str, Any] = {}
                support_rows: list[dict[str, Any]] = []
                if memory_parameters:
                    memory_summary, support_rows = trace_row(
                        model,
                        initial_memory,
                        previous_memory,
                        step=step,
                        stage=stage,
                        memory_bank_trainable=any(parameter.requires_grad for parameter in memory_parameters.values()),
                        probe_inputs=probe_inputs,
                    )
                    previous_memory = snapshot_memory_bank(model)
                memory_trace_rows.append({
                    "step": step,
                    "stage": stage,
                    "train_mse": train_evaluation.get("mse"),
                    "validation_mse": validation.get("mse"),
                    "test_mse": test_evaluation.get("mse"),
                    "train_mae": train_evaluation.get("mae"),
                    "validation_mae": validation.get("mae"),
                    "test_mae": test_evaluation.get("mae"),
                    **memory_summary,
                })
                for support_row in support_rows:
                    memory_support_rows.append({"task": task, "variant": spec["model_name"], **support_row})
                record["train_evaluation"] = train_evaluation
                record["test"] = test_evaluation
                record["memory"] = memory_summary
            history.append(record)
            if primary < best_metric:
                best_metric = primary
                best_validation = validation
                best_step = step
                atomic_torch_save(output_dir / "best_model.pt", {"model_state": model.state_dict(), "model_spec": spec, "resolved_config": resolved, "validation": validation, "step": step, "memory_protocol": memory_protocol, "memory_freeze_step": freeze_step})
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory = 0.0
    final_validation = _evaluate(model, validation_loader, device, task_type, int(run.get("eval_batches", 20)))
    final_train_evaluation = _evaluate(model, train_eval_loader, device=device, task_type=task_type, max_batches=int(run.get("eval_batches", 20))) if evaluate_train and train_eval_loader is not None else None
    final_test = _evaluate(model, test_loader, device=device, task_type=task_type, max_batches=int(run.get("eval_batches", 20))) if evaluate_test and test_loader is not None else None
    validation_descriptive: dict[str, float] | None = None
    validation_prediction_rows: list[dict[str, float]] = []
    test_descriptive: dict[str, float] | None = None
    test_prediction_rows: list[dict[str, float]] = []
    if task_type == "regression" and bool(run.get("save_validation_predictions", True)):
        prediction_tensor, target_tensor, validation_prediction_rows = _collect_regression_predictions(model, validation_loader, device, int(run.get("eval_batches", 20)))
        validation_descriptive = _descriptive_regression_metrics(prediction_tensor, target_tensor)
    if task_type == "regression" and evaluate_test and test_loader is not None and bool(run.get("save_test_predictions", False)):
        prediction_tensor, target_tensor, test_prediction_rows = _collect_regression_predictions(model, test_loader, device, int(run.get("eval_batches", 20)))
        test_descriptive = _descriptive_regression_metrics(prediction_tensor, target_tensor)
    final_checkpoint = output_dir / "final_model.pt"
    atomic_torch_save(final_checkpoint, {"model_state": model.state_dict(), "model_spec": spec, "resolved_config": resolved, "validation": final_validation, "test": final_test, "step": steps, "memory_protocol": memory_protocol, "memory_freeze_step": freeze_step})
    best_checkpoint_validation = None
    best_checkpoint_test = None
    best_checkpoint_test_descriptive = None
    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        best_checkpoint_validation = _evaluate(model, validation_loader, device, task_type, int(run.get("eval_batches", 20)))
        best_checkpoint_test = _evaluate(model, test_loader, device, task_type, int(run.get("eval_batches", 20))) if test_loader is not None else None
        if task_type == "regression" and evaluate_test and test_loader is not None and bool(run.get("save_test_predictions", False)):
            prediction_tensor, target_tensor, _ = _collect_regression_predictions(model, test_loader, device, int(run.get("eval_batches", 20)))
            best_checkpoint_test_descriptive = _descriptive_regression_metrics(prediction_tensor, target_tensor)
    metrics = {
        "run_id": run["run_id"], "task": task, "variant": spec["model_name"], "seed": seed,
        "device": str(device), "precision": precision, **capacity,
        "route_feature_dim": int(getattr(model, "route_feature_dim", -1)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_trainable_parameter_count": initial_memory_trainable_parameter_count,
        "memory_bank_parameter_count": memory_bank_parameter_count,
        "memory_protocol": memory_protocol, "memory_freeze_step": freeze_step,
        "memory_warmup_fraction": warmup_fraction, **variant_metadata,
        "training_steps": steps, "training_samples": steps * int(run.get("batch_size", 32)),
        "total_seconds": time.perf_counter() - start, "peak_memory_megabytes": peak_memory,
        "best_validation": best_validation or final_validation,
        "best_checkpoint_validation": best_checkpoint_validation,
        "best_checkpoint_test": best_checkpoint_test,
        "best_checkpoint_test_descriptive_metrics": best_checkpoint_test_descriptive,
        "final_validation": final_validation, "final_train_evaluation": final_train_evaluation, "final_test": final_test,
        "history": history, "status": "complete", "model_spec": spec, "data_metadata": data_meta,
        "best_checkpoint": str(output_dir / "best_model.pt"), "final_checkpoint": str(final_checkpoint),
        "memory_trace_points": len(memory_trace_rows),
    }
    primary_key = "mse" if task_type == "regression" else "cross_entropy"
    last_train = history[-1]["train"] if history else {}
    last_validation = history[-1]["validation"] if history else final_validation
    metrics["best_step"] = int(best_step or steps)
    metrics["history_points"] = len(history)
    metrics["tokens_or_samples_per_second"] = float(metrics["training_samples"] / max(metrics["total_seconds"], 1e-12))
    if primary_key in last_train and primary_key in last_validation:
        metrics["final_generalization_gap"] = float(last_validation[primary_key] - last_train[primary_key])
        metrics["final_train_metric"] = float(last_train[primary_key])
        metrics["final_validation_metric"] = float(last_validation[primary_key])
    if validation_descriptive is not None:
        metrics["validation_descriptive_metrics"] = validation_descriptive
    if test_descriptive is not None:
        metrics["test_descriptive_metrics"] = test_descriptive
    if memory_trace_rows:
        _write_rows(output_dir / "memory_training_trace.csv", memory_trace_rows)
    if memory_support_rows:
        _write_rows(output_dir / "memory_support_trace.csv", memory_support_rows)
    stale_failure = output_dir / "failure.json"
    if stale_failure.exists():
        stale_failure.unlink()
    save_json(output_dir / "metrics.json", metrics)
    with (output_dir / "stream_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        row = {"run_id": run["run_id"], "task": task, "variant": spec["model_name"], "seed": seed, **final_validation}
        if final_test:
            row.update({f"test_{key}": value for key, value in final_test.items()})
        if validation_descriptive is not None:
            row.update({f"descriptive_{key}": value for key, value in validation_descriptive.items()})
        if test_descriptive is not None:
            row.update({f"test_descriptive_{key}": value for key, value in test_descriptive.items()})
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if validation_prediction_rows:
        with (output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "target", "prediction", "error", "abs_error"])
            writer.writeheader()
            writer.writerows(validation_prediction_rows)
    if test_prediction_rows:
        with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "target", "prediction", "error", "abs_error"])
            writer.writeheader()
            writer.writerows(test_prediction_rows)
    return metrics

def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for kam-run-suite; install the project dependencies.") from error
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Suite config must contain a mapping at the top level.")
    return config


def run_suite(config: dict[str, Any], *, config_path: str | Path | None = None) -> list[dict[str, Any]]:
    output_root = Path(config.get("output_root", "results/phase2/runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(str(config.get("device", "auto")))
    precision = str(config.get("precision", "fp32"))
    registry = ExperimentRegistry(config.get("study_database", output_root / "study.sqlite"))
    completed: list[dict[str, Any]] = []
    runs = list(config.get("runs", []))
    if not runs:
        raise ValueError("Phase II suite configs must contain an explicit runs list.")
    manifest_rows: list[dict[str, Any]] = []
    for index, raw_run in enumerate(runs):
        run = dict(raw_run)
        run.setdefault("seed", int(config.get("seed", 7)))
        payload = json.dumps(run, sort_keys=True, default=str).encode()
        run_id = str(run.get("run_id") or hashlib.sha1(payload).hexdigest()[:12])
        run["run_id"] = run_id
        if registry.is_complete(run_id):
            metrics_path = output_root / run_id / "metrics.json"
            if metrics_path.exists():
                completed.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                stale_failure = output_root / run_id / "failure.json"
                if stale_failure.exists():
                    stale_failure.unlink()
            continue
        registry.start(run_id, str(config.get("suite_name", Path(config_path or "suite").stem)), run, run["seed"], str(output_root / run_id))
        try:
            metrics = _run_one(run, output_root, device, precision)
            registry.complete(run_id, metrics)
            completed.append(metrics)
            manifest_rows.append(metrics)
        except Exception as error:
            registry.fail(run_id, error)
            failure = {"run_id": run_id, "status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            save_json(output_root / run_id / "failure.json", failure)
            if bool(config.get("fail_fast", True)):
                registry.close()
                raise
    registry.close()
    metrics_table = Path(config.get("metrics_table", output_root / "all_metrics.csv"))
    if completed:
        fields = sorted({key for row in completed for key in row if not isinstance(row.get(key), (dict, list))})
        with metrics_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in completed:
                writer.writerow({key: row.get(key) for key in fields})
    save_json(output_root / "resolved_suite_config.json", {"config": config, "device": str(device), "precision": precision, "run_count": len(runs)})
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable Phase II experiment suite.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    results = run_suite(load_config(args.config), config_path=args.config)
    print(f"Completed {len(results)} Phase II runs.")


if __name__ == "__main__":
    main()
