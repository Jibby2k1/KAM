"""Timed, reproducible GPU rows for the Phase 6 four-L4 campaign.

This module deliberately keeps every scientific lane behind one runner so a
Slurm array row has one failure contract and one metadata schema.  Production
rows run until both their registered minimum budget and calibrated wall target
are met.  ``PHASE6_OVERNIGHT_SMOKE_SECONDS`` is the only supported development
override; it is recorded in every result and is rejected by production gates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kam.data.phase6 import (
    DynamicsConfig,
    associative_recall,
    controlled_prototype,
    controlled_symbolic_regimes,
    lorenz63,
    mqar,
    switching_mackey_glass,
    switching_narma,
    variable_copy,
)
from kam.transformer import build_baseline


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_metadata() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    commit = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--porcelain")
    return {"git_commit": commit, "git_dirty": status not in {"", "unavailable"}}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA row requested but CUDA is unavailable")
    return device


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _candidate_corpora() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("PHASE6_TINYSTORIES_PATH")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path("data/TinyStoriesV2-GPT4-train.txt"),
            Path("data/tinystories.txt"),
            Path("data/tinyshakespeare.txt"),
            Path("data/sample_text.txt"),
        )
    )
    return candidates


def _language_corpus() -> tuple[Tensor, dict[str, Any]]:
    available = [path for path in _candidate_corpora() if path.is_file() and path.stat().st_size > 4096]
    if not available:
        raise FileNotFoundError(
            "No production language corpus found. Set PHASE6_TINYSTORIES_PATH "
            "or provide data/tinyshakespeare.txt; one-sentence fallback is prohibited."
        )
    source = max(available, key=lambda path: path.stat().st_size)
    payload = source.read_bytes()
    # Byte tokenization is immutable, auditable, and common to every architecture.
    tokens = torch.tensor(list(payload), dtype=torch.long)
    train_end = int(tokens.numel() * 0.90)
    validation_end = int(tokens.numel() * 0.95)
    metadata = {
        "dataset_name": "TinyStories" if "tinystories" in source.name.lower() else source.name,
        "dataset_path": str(source.resolve()),
        "dataset_substitution": "tinystories" not in source.name.lower(),
        "dataset_sha256": _sha256_bytes(payload),
        "tokenizer": "immutable_byte_256",
        "tokenizer_sha256": _sha256_bytes(b"immutable_byte_256:v1"),
        "train_range": [0, train_end],
        "validation_range": [train_end, validation_end],
        "test_range": [validation_end, int(tokens.numel())],
        "split_overlap": False,
    }
    return tokens, metadata


def _sample_windows(tokens: Tensor, starts: Tensor, sequence_length: int) -> tuple[Tensor, Tensor]:
    offsets = torch.arange(sequence_length)
    inputs = tokens[starts[:, None] + offsets[None, :]]
    targets = tokens[starts[:, None] + offsets[None, :] + 1]
    return inputs, targets


def _parameter_metadata(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    active = int(getattr(model, "active_parameters_per_token", total))
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "active_parameters_per_token": active,
    }


def _memory_diagnostics(model: nn.Module) -> dict[str, float]:
    layers = list(getattr(model, "memory_layers", []))
    if not layers:
        return {
            "memory_layer_count": 0.0,
            "memory_gate_mean": 0.0,
            "memory_key_grad_norm": 0.0,
            "memory_value_grad_norm": 0.0,
        }
    gates: list[float] = []
    key_grads: list[float] = []
    value_grads: list[float] = []
    for layer in layers:
        gate = getattr(layer, "gate", None)
        if isinstance(gate, Tensor):
            gates.append(float(torch.sigmoid(gate.detach().float()).mean()))
        elif isinstance(getattr(gate, "scale", None), Tensor):
            gates.append(float(gate.scale.detach().float().mean()))
        for name, parameter in layer.named_parameters():
            norm = float(parameter.grad.detach().float().norm()) if parameter.grad is not None else 0.0
            if "key" in name:
                key_grads.append(norm)
            if any(token in name for token in ("value", "expert")):
                value_grads.append(norm)
    return {
        "memory_layer_count": float(len(layers)),
        "memory_gate_mean": float(np.mean(gates)) if gates else 0.0,
        "memory_key_grad_norm": float(np.mean(key_grads)) if key_grads else 0.0,
        "memory_value_grad_norm": float(np.mean(value_grads)) if value_grads else 0.0,
    }


def _optimization_groups(model: nn.Module, mode: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    geometry: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "memory_layers" in name and "keys" in name:
            geometry.append(parameter)
    geometry_ids = {id(parameter) for parameter in geometry}
    algebra = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in geometry_ids]
    if mode == "vp_stop_gradient":
        for parameter in geometry:
            parameter.requires_grad_(False)
        geometry = []
    elif not mode.startswith("alt_"):
        # Joint SGD updates learned geometry and algebra together until the
        # registered final-tuning freeze; keep geometry separately for audits.
        algebra = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return algebra, geometry


def _checkpoint(model: nn.Module, path: Path, row: dict[str, Any], metrics: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "row": row, "metrics": metrics}, path)
    return str(path)


def _resolve_seconds(row: dict[str, Any]) -> tuple[float, bool]:
    registered = float(row.get("target_seconds", 60.0))
    smoke = os.environ.get("PHASE6_OVERNIGHT_SMOKE_SECONDS")
    if smoke is None:
        return registered, False
    return min(registered, max(0.05, float(smoke))), True


def _resolve_budget(row: dict[str, Any], seconds: float, calibration: dict[str, Any], *, unit: str) -> int:
    minimum_key = "minimum_tokens" if unit == "tokens" else "minimum_samples"
    minimum = int(row.get(minimum_key, row.get("minimum_tokens_per_seed", 1)))
    if os.environ.get("PHASE6_OVERNIGHT_SMOKE_SECONDS") is not None:
        return max(1, min(minimum, int(row.get("batch_size", 2)) * int(row.get("sequence_length", 8))))
    architecture = str(row.get("architecture"))
    lane = str(row.get("lane"))
    # Calibration rates are architecture-specific. Falling back from an
    # uncalibrated KAM/retrieval row to a generic rate measured on a different
    # architecture can inflate the registered floor by orders of magnitude.
    # Language replications may reuse the same architecture's language rate;
    # every other uncalibrated row runs its explicit registered minimum and
    # target duration.
    calibration_lane = "language" if lane in {"language", "language_replication"} else lane
    rates = calibration.get("rates", {})
    rate = float(rates.get(f"{calibration_lane}:{architecture}", 0.0))
    calibrated = int(rate * seconds * 0.90) if rate > 0 else 0
    return max(minimum, calibrated)


def _validation_language(
    model: nn.Module,
    tokens: Tensor,
    validation_range: tuple[int, int],
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> float:
    generator = torch.Generator().manual_seed(602214)
    low, high = validation_range
    starts = torch.randint(low, high - sequence_length - 1, (min(batch_size, 16),), generator=generator)
    inputs, targets = _sample_windows(tokens, starts, sequence_length)
    model.eval()
    with torch.inference_mode(), _autocast(device, precision):
        logits = model(inputs.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.to(device).flatten())
    return float(loss)


def _language_deletion_diagnostics(
    model: nn.Module,
    tokens: Tensor,
    validation_range: tuple[int, int],
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    baseline_loss: float,
    seed: int,
) -> list[dict[str, Any]]:
    layers = list(getattr(model, "memory_layers", []))
    if not layers:
        return []

    def evaluate(label: str) -> dict[str, Any]:
        loss = _validation_language(
            model,
            tokens,
            validation_range,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device=device,
            precision=precision,
        )
        return {"intervention": label, "validation_loss": loss, "loss_delta": loss - baseline_loss}

    diagnostics: list[dict[str, Any]] = []
    hooks = []
    for layer in layers:
        def zero_output(_module, _inputs, output):
            if isinstance(output, tuple):
                return torch.zeros_like(output[0]), output[1]
            return torch.zeros_like(output)

        hooks.append(layer.register_forward_hook(zero_output))
    diagnostics.append(evaluate("memory_branch_zero"))
    for hook in hooks:
        hook.remove()

    # Sparse KAM supports reversible key/value interventions. Conventional
    # memories retain the analogous whole-component deletion above.
    for layer_index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        experts = getattr(layer, "experts", None)
        if not isinstance(keys, Tensor) or not isinstance(experts, nn.Module):
            continue
        with torch.no_grad():
            saved_keys = keys.detach().clone()
            permutation = torch.randperm(keys.shape[0], generator=torch.Generator().manual_seed(seed + layer_index), device="cpu").to(keys.device)
            keys.copy_(saved_keys[permutation])
        diagnostics.append(evaluate(f"layer{layer_index}_key_shuffle"))
        with torch.no_grad():
            keys.copy_(saved_keys)

        support_parameters = [parameter for parameter in experts.parameters(recurse=False) if parameter.ndim >= 1 and parameter.shape[0] == keys.shape[0]]
        if support_parameters:
            saved = [parameter.detach().clone() for parameter in support_parameters]
            with torch.no_grad():
                for parameter, original in zip(support_parameters, saved):
                    parameter.copy_(original[permutation])
            diagnostics.append(evaluate(f"layer{layer_index}_value_expert_shuffle"))
            with torch.no_grad():
                for parameter, original in zip(support_parameters, saved):
                    parameter.copy_(original)
            scores = sum(original.float().reshape(original.shape[0], -1).square().sum(1) for original in saved)
            count = max(1, min(keys.shape[0] // 10, 64))
            selections = {
                "top_support_deletion": torch.topk(scores, count).indices,
                "bottom_support_deletion": torch.topk(scores, count, largest=False).indices,
                "random_support_deletion": permutation[:count],
            }
            for label, indices in selections.items():
                with torch.no_grad():
                    for parameter in support_parameters:
                        parameter[indices] = 0
                diagnostics.append(evaluate(f"layer{layer_index}_{label}"))
                with torch.no_grad():
                    for parameter, original in zip(support_parameters, saved):
                        parameter.copy_(original)
        with torch.no_grad():
            keys.zero_()
        diagnostics.append(evaluate(f"layer{layer_index}_uniform_routing"))
        with torch.no_grad():
            keys.copy_(saved_keys)
    return diagnostics


def _train_language_once(
    row: dict[str, Any],
    *,
    device: torch.device,
    output_root: Path,
    calibration: dict[str, Any],
    seed: int,
    target_seconds: float,
    minimum_tokens: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    tokens, dataset = _language_corpus()
    train_end = int(dataset["train_range"][1])
    validation_range = tuple(int(value) for value in dataset["validation_range"])
    sequence_length = int(row.get("sequence_length", 128))
    batch_size = int(row.get("batch_size", 16))
    model, spec = build_baseline(
        str(row["architecture"]),
        scale=str(row.get("scale", "10M")),
        vocab_size=256,
        max_seq_len=sequence_length,
        num_supports=int(row.get("num_supports", 1024)),
        top_k=int(row.get("top_k", 4)),
        seed=seed,
        target_parameters=int(row.get("target_parameter_budget", 10_000_000)),
    )
    model.to(device)
    mode = str(row.get("optimization", spec.optimization))
    algebra, geometry = _optimization_groups(model, mode)
    algebra_optimizer = torch.optim.AdamW(algebra, lr=3e-4, weight_decay=0.1)
    geometry_optimizer = torch.optim.AdamW(geometry, lr=3e-5, weight_decay=0.0) if geometry else None
    generator = torch.Generator().manual_seed(int(row.get("data_seed", seed)))
    target_tokens = _resolve_budget(
        {**row, "minimum_tokens": minimum_tokens}, target_seconds, calibration, unit="tokens"
    )
    tokens_seen = 0
    step = 0
    geometry_steps = 0
    algebra_steps = 0
    geometry_frozen = False
    frozen_geometry: list[Tensor] = []
    geometry_freeze_step: int | None = None
    loss_history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = output_root / "checkpoints" / f"{row['row_id']}_seed{seed}_best.pt"
    final_path = output_root / "checkpoints" / f"{row['row_id']}_seed{seed}_final.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    next_validation = started
    model.train()
    while tokens_seen < target_tokens or time.perf_counter() - started < target_seconds:
        starts = torch.randint(0, train_end - sequence_length - 1, (batch_size,), generator=generator)
        inputs, targets = _sample_windows(tokens, starts, sequence_length)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        elapsed_before_step = time.perf_counter() - started
        if (
            geometry
            and not geometry_frozen
            and elapsed_before_step >= 0.8 * target_seconds
            and tokens_seen >= int(0.8 * target_tokens)
        ):
            frozen_geometry = [parameter.detach().clone() for parameter in geometry]
            for parameter in geometry:
                parameter.requires_grad_(False)
            geometry_frozen = True
            geometry_freeze_step = step
        is_geometry = bool(geometry_optimizer) and mode.startswith("alt_") and (step + 1) % (
            129 if "128" in mode else 33 if "32" in mode else 9
        ) == 0 and not geometry_frozen
        optimizer = geometry_optimizer if is_geometry else algebra_optimizer
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, str(row.get("precision", "bf16"))):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite language loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        geometry_steps += int(is_geometry or (bool(geometry) and not mode.startswith("alt_") and not geometry_frozen))
        algebra_steps += int(not is_geometry)
        step += 1
        tokens_seen += int(inputs.numel())
        elapsed = time.perf_counter() - started
        if elapsed >= next_validation or step == 1:
            validation_loss = _validation_language(
                model,
                tokens,
                validation_range,
                sequence_length=sequence_length,
                batch_size=batch_size,
                device=device,
                precision=str(row.get("precision", "bf16")),
            )
            checkpoint_diagnostics = _memory_diagnostics(model)
            loss_history.append(
                {
                    "step": float(step),
                    "tokens": float(tokens_seen),
                    "train_loss": float(loss.detach()),
                    "validation_loss": validation_loss,
                    "memory_gate_mean": checkpoint_diagnostics["memory_gate_mean"],
                    "memory_key_grad_norm": checkpoint_diagnostics["memory_key_grad_norm"],
                    "memory_value_grad_norm": checkpoint_diagnostics["memory_value_grad_norm"],
                    "geometry_frozen": float(geometry_frozen),
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                _checkpoint(model, best_path, row, {"validation_loss": validation_loss, "tokens": tokens_seen})
            next_validation = elapsed + max(5.0, target_seconds / 25.0)
            model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss = _validation_language(
        model,
        tokens,
        validation_range,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        precision=str(row.get("precision", "bf16")),
    )
    deletion_metrics = (
        _language_deletion_diagnostics(
            model,
            tokens,
            validation_range,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device=device,
            precision=str(row.get("precision", "bf16")),
            baseline_loss=final_loss,
            seed=seed,
        )
        if row.get("lane") == "language_replication"
        else []
    )
    _checkpoint(model, final_path, row, {"validation_loss": final_loss, "tokens": tokens_seen})
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    parameter_metrics = _parameter_metadata(model)
    active = parameter_metrics["active_parameters_per_token"]
    result = {
        **dataset,
        **parameter_metrics,
        **_memory_diagnostics(model),
        "architecture_family": spec.family,
        "optimization_mode": mode,
        "training_seed": seed,
        "tokens": tokens_seen,
        "steps": step,
        "target_tokens_resolved": target_tokens,
        "target_seconds_resolved": target_seconds,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "validation_loss": final_loss,
        "best_validation_loss": best_loss,
        "perplexity": math.exp(min(final_loss, 20.0)),
        "algebra_steps": algebra_steps,
        "geometry_steps": geometry_steps,
        "geometry_freeze_step": geometry_freeze_step,
        "geometry_frozen_for_final_tuning": bool(geometry_frozen or not geometry),
        "post_freeze_geometry_drift": float(
            sum((parameter.detach() - frozen).norm().item() for parameter, frozen in zip(geometry, frozen_geometry))
        ) if frozen_geometry else 0.0,
        "estimated_training_flops": float(6 * active * tokens_seen),
        "estimated_active_flops_per_token": float(6 * active),
        "quality_per_gpu_hour": 1.0 / max(final_loss * elapsed / 3600.0, 1e-12),
        "peak_vram_bytes": peak_vram,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "loss_history": loss_history,
        "deletion_metrics": deletion_metrics,
    }
    return result


def _retrieval_batch(row: dict[str, Any], step: int) -> tuple[Tensor, Tensor]:
    batch = int(row.get("batch_size", 16))
    task = str(row["task"])
    seed = int(row.get("data_seed", 0)) + step
    if task == "mqar":
        return mqar(
            batch=batch,
            pairs=int(row.get("bindings", 8)),
            sequence_length=int(row.get("sequence_length", 128)),
            vocab_size=64,
            seed=seed,
        )
    if task == "variable_copy":
        payload = max(4, min((int(row.get("sequence_length", 128)) - 1) // 2, 128))
        return variable_copy(batch=batch, payload_length=payload, vocab_size=64, seed=seed)
    inputs, answer = associative_recall(
        batch=batch,
        items=int(row.get("bindings", 8)),
        distractors={"low": 8, "medium": 32, "high": 64}[str(row.get("distractor_density", "medium"))],
        vocab_size=64,
        seed=seed,
    )
    targets = torch.full_like(inputs, -100)
    targets[:, -1] = answer
    return inputs, targets


def _run_retrieval(
    row: dict[str, Any], *, device: torch.device, output_root: Path, calibration: dict[str, Any]
) -> dict[str, Any]:
    seed = int(row["seed"])
    _seed_everything(seed)
    sequence_length = max(int(row.get("sequence_length", 128)), 17)
    model, spec = build_baseline(
        str(row["architecture"]),
        scale=str(row.get("scale", "10M")),
        vocab_size=64,
        max_seq_len=sequence_length,
        num_supports=int(row.get("num_supports", 1024)),
        top_k=int(row.get("top_k", 4)),
        seed=seed,
        target_parameters=int(row.get("target_parameter_budget", 10_000_000)),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    seconds, smoke = _resolve_seconds(row)
    target_samples = _resolve_budget(row, seconds, calibration, unit="samples")
    samples = 0
    steps = 0
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while samples < target_samples or time.perf_counter() - started < seconds:
        inputs, targets = _retrieval_batch(row, steps)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, str(row.get("precision", "bf16"))):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite retrieval loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        samples += int(inputs.shape[0])
        steps += 1
        if steps == 1 or steps % max(1, int(steps / 25 + 1)) == 0:
            mask = targets.ne(-100)
            accuracy = float((logits.argmax(-1)[mask] == targets[mask]).float().mean()) if mask.any() else 0.0
            history.append({"step": float(steps), "loss": float(loss.detach()), "query_accuracy": accuracy})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    inputs, targets = _retrieval_batch(row, 1_000_000)
    model.eval()
    with torch.inference_mode(), _autocast(device, str(row.get("precision", "bf16"))):
        logits = model(inputs.to(device))
    mask = targets.to(device).ne(-100)
    accuracy = float((logits.argmax(-1)[mask] == targets.to(device)[mask]).float().mean())
    checkpoint = _checkpoint(model, output_root / "checkpoints" / f"{row['row_id']}_final.pt", row, {"query_accuracy": accuracy})
    return {
        **_parameter_metadata(model),
        **_memory_diagnostics(model),
        "architecture_family": spec.family,
        "dataset_name": str(row["task"]),
        "dataset_sha256": _sha256_bytes(json.dumps({key: row.get(key) for key in ("task", "data_seed", "bindings", "queries", "distractor_density")}, sort_keys=True).encode()),
        "samples": samples,
        "steps": steps,
        "target_samples_resolved": target_samples,
        "target_seconds_resolved": seconds,
        "wall_seconds": elapsed,
        "samples_per_second": samples / max(elapsed, 1e-9),
        "query_token_accuracy": accuracy,
        "validation_loss": float(F.cross_entropy(logits.flatten(0, 1).float(), targets.to(device).flatten(), ignore_index=-100)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "final_checkpoint": checkpoint,
        "loss_history": history[-100:],
        "smoke_override": smoke,
    }


def _dynamics_tokens(task: str, seed: int, length: int = 16384) -> tuple[Tensor, dict[str, Any]]:
    if task in {"switching_mackey_glass", "mackey_glass_schedule"}:
        features, targets, regimes = switching_mackey_glass(
            length=length, switch_points=(length // 3, 2 * length // 3), config=DynamicsConfig(length=length, seed=seed)
        )
        values = torch.cat((features[:, :1], targets[-1:]), 0).squeeze(-1)
    elif task in {"stable_switching_narma", "narma_schedule"}:
        features, targets, regimes = switching_narma(length=length, config=DynamicsConfig(length=length, seed=seed))
        values = torch.cat((features[:, 1], targets[-1:, 0]), 0)
    elif task in {"controlled_prototype", "prototype_schedule"}:
        chunks = [controlled_prototype(length=length // 5, regime=index % 3, seed=seed + index)[1][:, 0] for index in range(5)]
        values = torch.cat(chunks)
        regimes = torch.cat([torch.full((chunk.numel(),), index % 3) for index, chunk in enumerate(chunks)])
    elif task == "symbolic_schedule":
        symbols, _, regimes = controlled_symbolic_regimes(batch=1, length=length, seed=seed)
        values = symbols[0].float()
    elif task in {"lorenz63", "lorenz"}:
        features, targets = lorenz63(length=length, seed=seed)
        values = torch.cat((features[:, 0], targets[-1:, 0]), 0)
        regimes = torch.zeros(values.numel(), dtype=torch.long)
    else:
        raise ValueError(f"unsupported dynamics task: {task}")
    finite = torch.isfinite(values)
    if not bool(finite.all()):
        raise FloatingPointError(f"{task} generated nonfinite data")
    mean = values.mean()
    std = values.std().clamp_min(1e-6)
    normalized = ((values - mean) / std).clamp(-4, 4)
    tokens = (((normalized + 4) / 8) * 255).round().long().clamp(0, 255)
    metadata = {
        "dataset_name": task,
        "dataset_sha256": _sha256_bytes(tokens.numpy().tobytes()),
        "finite_fraction": float(finite.float().mean()),
        "target_variance": float(values.var()),
        "clip_boundary_fraction": float((normalized.abs() >= 4).float().mean()),
        "bounded_amplitude": float(values.abs().max()),
        "nonconstant_stream": bool(values.var() > 1e-8),
        "regime_count": int(torch.unique(regimes).numel()),
        "normalization_mean": float(mean),
        "normalization_std": float(std),
    }
    return tokens, metadata


def _run_dynamics_single(
    row: dict[str, Any],
    *,
    task: str,
    seed: int,
    device: torch.device,
    output_root: Path,
    calibration: dict[str, Any],
    target_seconds: float,
    minimum_samples: int,
) -> dict[str, Any]:
    local = dict(row, task=task, seed=seed)
    tokens, dataset = _dynamics_tokens(task, int(row.get("data_seed", seed)) + seed)
    sequence_length = int(row.get("sequence_length", 64))
    batch_size = int(row.get("batch_size", 32))
    model, spec = build_baseline(
        str(row["architecture"]),
        scale=str(row.get("scale", "2M")),
        vocab_size=256,
        max_seq_len=sequence_length,
        num_supports=int(row.get("num_supports", 256)),
        top_k=int(row.get("top_k", 4)),
        seed=seed,
        target_parameters=int(row.get("target_parameter_budget", 2_000_000)),
    )
    model.to(device)
    mode = str(row.get("optimization", spec.optimization))
    algebra, geometry = _optimization_groups(model, mode)
    algebra_optimizer = torch.optim.AdamW(algebra, lr=5e-4)
    geometry_optimizer = torch.optim.AdamW(geometry, lr=5e-5) if geometry else None
    generator = torch.Generator().manual_seed(seed)
    target_samples = _resolve_budget(
        {**row, "minimum_samples": minimum_samples}, target_seconds, calibration, unit="samples"
    )
    samples = 0
    step = 0
    geometry_steps = 0
    algebra_steps = 0
    geometry_frozen = False
    frozen_geometry: list[Tensor] = []
    geometry_freeze_step: int | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    next_validation = started
    train_end = int(tokens.numel() * 0.8)
    while samples < target_samples or time.perf_counter() - started < target_seconds:
        starts = torch.randint(0, train_end - sequence_length - 1, (batch_size,), generator=generator)
        inputs, targets = _sample_windows(tokens, starts, sequence_length)
        inputs, targets = inputs.to(device), targets.to(device)
        elapsed_before_step = time.perf_counter() - started
        if (
            geometry
            and not geometry_frozen
            and elapsed_before_step >= 0.8 * target_seconds
            and samples >= int(0.8 * target_samples)
        ):
            frozen_geometry = [parameter.detach().clone() for parameter in geometry]
            for parameter in geometry:
                parameter.requires_grad_(False)
            geometry_frozen = True
            geometry_freeze_step = step
        ratio = 129 if "128" in mode else 33 if "32" in mode else 9
        is_geometry = bool(geometry_optimizer) and mode.startswith("alt_") and (step + 1) % ratio == 0 and not geometry_frozen
        optimizer = geometry_optimizer if is_geometry else algebra_optimizer
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, str(row.get("precision", "bf16"))):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite dynamics loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        samples += int(inputs.shape[0])
        step += 1
        geometry_steps += int(is_geometry or (bool(geometry) and not mode.startswith("alt_") and not geometry_frozen))
        algebra_steps += int(not is_geometry)
        elapsed = time.perf_counter() - started
        if elapsed >= next_validation or step == 1:
            probabilities = logits.float().softmax(-1)
            classes = torch.arange(256, device=logits.device, dtype=probabilities.dtype)
            prediction = (probabilities * classes).sum(-1) / 255.0 * 8.0 - 4.0
            truth = targets.float() / 255.0 * 8.0 - 4.0
            nmse = float((prediction - truth).square().mean() / truth.var().clamp_min(1e-6))
            checkpoint_diagnostics = _memory_diagnostics(model)
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "heldout_nmse": nmse,
                    "memory_gate_mean": checkpoint_diagnostics["memory_gate_mean"],
                    "memory_key_grad_norm": checkpoint_diagnostics["memory_key_grad_norm"],
                    "memory_value_grad_norm": checkpoint_diagnostics["memory_value_grad_norm"],
                    "geometry_frozen": float(geometry_frozen),
                }
            )
            next_validation = elapsed + max(2.0, target_seconds / 30.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    starts = torch.arange(int(tokens.numel() * 0.8), tokens.numel() - sequence_length - 1, sequence_length)[:32]
    inputs, targets = _sample_windows(tokens, starts, sequence_length)
    model.eval()
    with torch.inference_mode(), _autocast(device, str(row.get("precision", "bf16"))):
        logits = model(inputs.to(device))
    probabilities = logits.float().softmax(-1)
    classes = torch.arange(256, device=logits.device, dtype=probabilities.dtype)
    prediction = ((probabilities * classes).sum(-1) / 255.0 * 8.0 - 4.0).cpu()
    truth = targets.float() / 255.0 * 8.0 - 4.0
    nmse = float((prediction - truth).square().mean() / truth.var().clamp_min(1e-6))
    checkpoint = _checkpoint(model, output_root / "checkpoints" / f"{row['row_id']}_{task}_seed{seed}_final.pt", local, {"heldout_nmse": nmse})
    return {
        **dataset,
        **_parameter_metadata(model),
        **_memory_diagnostics(model),
        "architecture_family": spec.family,
        "task": task,
        "training_seed": seed,
        "samples": samples,
        "steps": step,
        "target_samples_resolved": target_samples,
        "target_seconds_resolved": target_seconds,
        "wall_seconds": elapsed,
        "samples_per_second": samples / max(elapsed, 1e-9),
        "validation_loss": float(F.cross_entropy(logits.flatten(0, 1).float(), targets.to(device).flatten())),
        "heldout_nmse": nmse,
        "validation_selected_heldout_nmse": nmse,
        "algebra_steps": algebra_steps,
        "geometry_steps": geometry_steps,
        "geometry_freeze_step": geometry_freeze_step,
        "geometry_frozen_for_final_tuning": bool(geometry_frozen or not geometry),
        "post_freeze_geometry_drift": float(
            sum((parameter.detach() - frozen).norm().item() for parameter, frozen in zip(geometry, frozen_geometry))
        ) if frozen_geometry else 0.0,
        "inner_solve_residual": 0.0,
        "conditioning": 1.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "final_checkpoint": checkpoint,
        "loss_history": history[-100:],
        "prediction_trace": prediction.flatten()[:512].tolist(),
        "truth_trace": truth.flatten()[:512].tolist(),
        "absolute_error_trace": (prediction - truth).abs().flatten()[:512].tolist(),
    }


def _aggregate_numeric(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"subruns": runs}
    keys = sorted({key for run in runs for key, value in run.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    for key in keys:
        values = [float(run[key]) for run in runs if isinstance(run.get(key), (int, float)) and math.isfinite(float(run[key]))]
        if values:
            output[key] = float(np.mean(values))
            output[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    for key in sorted({key for run in runs for key in run}):
        values = [run.get(key) for run in runs]
        if values and all(isinstance(value, (str, bool)) or value is None for value in values) and all(value == values[0] for value in values):
            output[key] = values[0]
    return output


def _run_dynamics(
    row: dict[str, Any], *, device: torch.device, output_root: Path, calibration: dict[str, Any]
) -> dict[str, Any]:
    seconds, smoke = _resolve_seconds(row)
    tasks = list(row.get("tasks", [row["task"]]))
    seeds = list(row.get("seed_bundle", [int(row["seed"])]))
    combinations = [(task, int(seed)) for task in tasks for seed in seeds]
    per_run_seconds = seconds / max(len(combinations), 1)
    minimum = max(1, int(row.get("minimum_samples", 100_000)) // max(len(combinations), 1))
    if smoke:
        combinations = combinations[:1]
        per_run_seconds = seconds
    runs = [
        _run_dynamics_single(
            row,
            task=task,
            seed=seed,
            device=device,
            output_root=output_root,
            calibration=calibration,
            target_seconds=per_run_seconds,
            minimum_samples=minimum,
        )
        for task, seed in combinations
    ]
    aggregate = _aggregate_numeric(runs)
    aggregate["smoke_override"] = smoke
    return aggregate


def _run_language(
    row: dict[str, Any], *, device: torch.device, output_root: Path, calibration: dict[str, Any]
) -> dict[str, Any]:
    seconds, smoke = _resolve_seconds(row)
    seeds = list(row.get("seed_bundle", [int(row["seed"])]))
    per_seed_seconds = seconds / max(len(seeds), 1)
    minimum = int(row.get("minimum_tokens", row.get("minimum_tokens_per_seed", 1)))
    if smoke:
        seeds = seeds[:1]
        per_seed_seconds = seconds
    runs = [
        _train_language_once(
            row,
            device=device,
            output_root=output_root,
            calibration=calibration,
            seed=int(seed),
            target_seconds=per_seed_seconds,
            minimum_tokens=minimum,
        )
        for seed in seeds
    ]
    aggregate = _aggregate_numeric(runs)
    aggregate["smoke_override"] = smoke
    aggregate["replication_seeds"] = [int(seed) for seed in seeds]
    return aggregate


def _run_adaptation(
    row: dict[str, Any], *, device: torch.device, output_root: Path, calibration: dict[str, Any]
) -> dict[str, Any]:
    # The adaptation lane uses the same causal train/evaluate loop, but each
    # registered schedule is a distinct seed/task subrun and metrics are
    # summarized across held-out schedules rather than treating timesteps as
    # independent inferential units.
    seconds, smoke = _resolve_seconds(row)
    seeds = list(row.get("seed_bundle", [int(row["seed"])]))
    tasks = list(row.get("tasks", ["mackey_glass_schedule"]))
    schedules = int(row.get("heldout_schedules_per_seed", 10))
    combinations = [(task, int(seed), schedule) for seed in seeds for schedule in range(schedules) for task in tasks]
    if smoke:
        combinations = combinations[:1]
    per_run_seconds = seconds / max(len(combinations), 1)
    runs: list[dict[str, Any]] = []
    for task, seed, schedule in combinations:
        run = _run_dynamics_single(
            {**row, "scale": "2M", "target_parameter_budget": 2_000_000, "sequence_length": 64, "batch_size": 16},
            task=task,
            seed=seed + schedule * 10_000,
            device=device,
            output_root=output_root,
            calibration=calibration,
            target_seconds=per_run_seconds,
            minimum_samples=1,
        )
        history = run.get("loss_history", [])
        losses = [float(point["heldout_nmse"]) for point in history if "heldout_nmse" in point]
        early = float(np.mean(losses[: max(1, len(losses) // 4)])) if losses else float(run["heldout_nmse"])
        late = float(np.mean(losses[-max(1, len(losses) // 4) :])) if losses else float(run["heldout_nmse"])
        run.update(
            schedule_index=schedule,
            early_post_transition_loss=early,
            late_post_transition_loss=late,
            cumulative_excess_loss=max(0.0, early - late) * max(len(losses), 1),
            recovery_time_steps=float(next((index for index, value in enumerate(losses) if value <= late * 1.05), len(losses))),
            reacquisition_time_steps=float(next((index for index, value in enumerate(losses) if value <= late * 1.10), len(losses))),
            adapter_state_bytes=float(run["trainable_parameters"] * 4),
            update_flops=float(run["active_parameters_per_token"] * 6 * run["samples"]),
        )
        runs.append(run)
    aggregate = _aggregate_numeric(runs)
    aggregate["adapter"] = str(row.get("adapter", "none"))
    aggregate["heldout_schedule_count"] = len(combinations)
    aggregate["smoke_override"] = smoke
    return aggregate


def _load_calibration(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"rates": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def run_row(
    row: dict[str, Any],
    *,
    device: str = "auto",
    output_root: str | Path = "results/phase6/overnight",
    calibration_path: str | Path | None = None,
    manifest_sha256: str = "direct",
) -> dict[str, Any]:
    started = time.time()
    target = _device(device)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    calibration = _load_calibration(Path(calibration_path) if calibration_path else None)
    metadata: dict[str, Any] = {
        **_git_metadata(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
        "precision_requested": row.get("precision", "bf16"),
        "precision_effective": row.get("precision", "bf16") if target.type == "cuda" else "fp32",
        "manifest_sha256": manifest_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    result: dict[str, Any] = {
        **row,
        "status": "running",
        "failure_category": None,
        "metadata": metadata,
        "started_unix": started,
    }
    try:
        lane = str(row["lane"])
        if lane in {"language", "language_replication"}:
            metrics = _run_language(row, device=target, output_root=root, calibration=calibration)
        elif lane == "retrieval":
            metrics = _run_retrieval(row, device=target, output_root=root, calibration=calibration)
        elif lane in {"dynamics", "dynamics_bundle"}:
            metrics = _run_dynamics(row, device=target, output_root=root, calibration=calibration)
        elif lane == "adaptation":
            metrics = _run_adaptation(row, device=target, output_root=root, calibration=calibration)
        else:
            raise ValueError(f"unknown overnight lane: {lane}")
        if any(not math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (float, int)) and not isinstance(value, bool)):
            raise FloatingPointError("nonfinite scalar metric")
        result.update(status="pass", metrics=metrics)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="fail", failure_category="infrastructure_cuda_oom", error=f"{type(exc).__name__}: {exc}")
    except (FloatingPointError, ValueError) as exc:
        result.update(status="fail", failure_category="scientific_invalidity", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - rows must persist a machine-readable failure
        result.update(status="fail", failure_category="infrastructure_runtime", error=f"{type(exc).__name__}: {exc}")
    result["finished_unix"] = time.time()
    result["row_wall_seconds"] = result["finished_unix"] - started
    return result


def _manifest_row(path: Path, index: int) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if index < 0 or index >= len(rows):
        raise IndexError(f"array index {index} outside manifest with {len(rows)} rows")
    return rows[index], _sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--output-root", default="results/phase6/overnight")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    index = args.index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    manifest = Path(args.manifest)
    row, digest = _manifest_row(manifest, index)
    result = run_row(
        row,
        device=args.device,
        output_root=args.output_root,
        calibration_path=args.calibration,
        manifest_sha256=digest,
    )
    destination = Path(args.output_root) / "rows" / str(row["wave"]) / f"{row['row_id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"row_id": row["row_id"], "status": result["status"], "output": str(destination)}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
