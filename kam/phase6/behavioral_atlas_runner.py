"""L4-oriented Stage 0 runner for the Phase 6.2 behavioral atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kam.memory import SparseMemoryConfig, SparseSeparableMemory
from kam.phase6.behavioral_atlas_instrumentation import (
    AnchorReference,
    WindowDynamicsAccumulator,
    build_anchor_bank,
    evaluate_anchor_behavior,
    exact_step_update_metrics,
    matched_key_expert_permutation_check,
    optimizer_state_norms,
)
from kam.phase6.overnight_runner import _autocast, _language_corpus, _sample_windows, _validation_language
from kam.phase6.parameter_dynamics_runner import (
    _save_key_snapshot,
    _save_model_snapshot,
    _seed_everything,
    _set_trainable,
    _validation_with_probes,
    configure_l4_execution,
)
from kam.phase6.parameter_trace import ParameterTraceRecorder, key_hash, state_hash
from kam.transformer.config import TransformerConfig
from kam.transformer.decoder import ModernDecoder


def build_behavioral_atlas_model(row: dict[str, Any]) -> ModernDecoder:
    config = TransformerConfig(
        d_model=int(row["d_model"]),
        n_heads=int(row["n_heads"]),
        n_layers=int(row["n_layers"]),
        d_ff=int(row["d_ff"]),
        vocab_size=int(row.get("vocab_size", 256)),
        max_seq_len=int(row["sequence_length"]),
    )
    if str(row.get("architecture")) == "T0":
        return ModernDecoder(config)
    fixed = not bool(row.get("geometry_trainable", True))
    layers = [
        SparseSeparableMemory(
            SparseMemoryConfig(
                d_model=config.d_model,
                num_supports=int(row["num_supports"]),
                top_k=int(row["top_k"]),
                expert_mode=str(row.get("expert_mode", "low_rank")),
                expert_rank=int(row.get("expert_rank", 4)),
                geometry_mode="fixed_random" if fixed else "learned_full",
                metric=str(row.get("router_metric", "dot")),
                temperature=float(row.get("router_temperature", 1.0)),
            ),
            seed=int(row["seed"]) + index,
        )
        for index in range(config.n_layers)
    ]
    return ModernDecoder(config, memory_layers=layers)


def _optimizer(parameters, *, lr: float, weight_decay: float, device: torch.device):
    values = list(parameters)
    if not values:
        return None
    options: dict[str, Any] = {"lr": float(lr), "weight_decay": float(weight_decay)}
    if device.type == "cuda":
        options["fused"] = True
    return torch.optim.AdamW(values, **options)


def _joint_optimizer(algebra, geometry, row: dict[str, Any], device: torch.device):
    groups = [{"params": list(algebra), "lr": float(row["algebra_lr"]), "weight_decay": float(row["algebra_weight_decay"]), "group_name": "algebra"}]
    if geometry:
        groups.append({"params": list(geometry), "lr": float(row["geometry_lr"]), "weight_decay": float(row["geometry_weight_decay"]), "group_name": "geometry"})
    options: dict[str, Any] = {}
    if device.type == "cuda":
        options["fused"] = True
    return torch.optim.AdamW(groups, **options)


def _optimizer_provenance(
    row: dict[str, Any],
    optimizers: list[torch.optim.Optimizer | None],
    *,
    compiled: bool,
) -> dict[str, Any]:
    groups = []
    for optimizer_index, optimizer in enumerate(optimizers):
        if optimizer is None:
            continue
        for group_index, group in enumerate(optimizer.param_groups):
            groups.append({
                "optimizer_index": optimizer_index,
                "group_index": group_index,
                "group_name": group.get("group_name", "algebra" if optimizer_index == 0 else "geometry"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
            })
    return {
        "declared_label": str(row["optimization"]),
        "effective_optimizer_class": "AdamW",
        "effective_schedule": str(row["optimization"]),
        "alternating_ratio": row.get("alternating_ratio"),
        "compiled_training_forward": bool(compiled),
        "parameter_groups": groups,
        "label_matches_executable": "sgd" not in str(row["optimization"]).lower(),
    }


def _training_start_schedule(
    *,
    train_end: int,
    sequence_length: int,
    batch_size: int,
    target_tokens: int,
    seed: int,
) -> tuple[Tensor, str]:
    tokens_per_step = batch_size * sequence_length
    steps = math.ceil(target_tokens / tokens_per_step)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    starts = torch.randint(0, train_end - sequence_length - 1, (steps, batch_size), generator=generator)
    digest = hashlib.sha256()
    digest.update(str(tuple(starts.shape)).encode())
    digest.update(starts.numpy().tobytes())
    return starts, digest.hexdigest()


def _compile_training_forward(model: nn.Module, enabled: bool, device: torch.device) -> tuple[nn.Module, bool, str | None]:
    if not enabled:
        return model, False, None
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return model, False, "compile_unavailable"
    try:
        return torch.compile(model, mode="reduce-overhead"), True, None
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        return model, False, f"compile_setup_failed:{type(exc).__name__}:{exc}"


def run_behavioral_atlas_row(row: dict[str, Any], *, device: str | torch.device, output_root: str | Path) -> dict[str, Any]:
    device = torch.device(device)
    output_root = Path(output_root)
    seed = int(row["seed"])
    _seed_everything(seed)
    execution = configure_l4_execution(device)
    tokens, dataset = _language_corpus(row)
    train_end = int(dataset["train_range"][1])
    validation_range = tuple(int(value) for value in dataset["validation_range"])
    test_range = tuple(int(value) for value in dataset["test_range"])
    sequence_length = int(row["sequence_length"])
    batch_size = int(row["batch_size"])
    precision = str(row.get("precision", "bf16"))
    model = build_behavioral_atlas_model(row).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    target_budget = row.get("target_parameter_budget")
    if target_budget is not None:
        tolerance = float(row.get("parameter_tolerance_fraction", 0.01))
        if abs(total_parameters - int(target_budget)) / max(int(target_budget), 1) > tolerance:
            raise AssertionError("model is outside the registered parameter tolerance")
    initial_state_hash = state_hash(model)
    initial_key_hash = key_hash(model)
    geometry = [layer.keys for layer in model.memory_layers]
    geometry_ids = {id(parameter) for parameter in geometry}
    algebra = [parameter for parameter in model.parameters() if id(parameter) not in geometry_ids]
    geometry_trainable = bool(row.get("geometry_trainable", True)) and bool(geometry)
    fixed = not geometry_trainable
    mode = str(row["optimization"])
    if fixed:
        _set_trainable(algebra, geometry, algebra_active=True, geometry_active=False)
    if mode == "alternating_adamw":
        joint_optimizer = None
        algebra_optimizer = _optimizer(algebra, lr=float(row["algebra_lr"]), weight_decay=float(row["algebra_weight_decay"]), device=device)
        geometry_optimizer = _optimizer(geometry, lr=float(row["geometry_lr"]), weight_decay=float(row["geometry_weight_decay"]), device=device)
        if algebra_optimizer is not None:
            algebra_optimizer.param_groups[0]["group_name"] = "algebra"
        if geometry_optimizer is not None:
            geometry_optimizer.param_groups[0]["group_name"] = "geometry"
    else:
        joint_optimizer = _joint_optimizer(algebra, geometry if geometry_trainable else [], row, device)
        algebra_optimizer = geometry_optimizer = None
    optimizers = [joint_optimizer, algebra_optimizer, geometry_optimizer]
    training_model, compile_applied, compile_error = _compile_training_forward(model, bool(row.get("compile_training", False)), device)
    execution.update({"compile_requested": bool(row.get("compile_training", False)), "compile_applied": compile_applied, "compile_error": compile_error})
    optimizer_provenance = _optimizer_provenance(row, optimizers, compiled=compile_applied)

    smoke_tokens = os.environ.get("PHASE6_BEHAVIORAL_ATLAS_SMOKE_TOKENS")
    target_tokens = min(int(row["target_tokens"]), int(smoke_tokens)) if smoke_tokens else int(row["target_tokens"])
    checkpoints = sorted({int(value) for value in row["validation_token_checkpoints"] if 0 <= int(value) <= target_tokens} | {0, target_tokens})
    freeze_fraction = float(row.get("freeze_fraction", 1.0))
    freeze_target = int(freeze_fraction * target_tokens)
    starts_schedule, sample_order_hash = _training_start_schedule(
        train_end=train_end,
        sequence_length=sequence_length,
        batch_size=batch_size,
        target_tokens=target_tokens,
        seed=int(row["data_seed"]),
    )
    trace_level = str(row.get("trace_level", "standard"))
    anchor_bank = build_anchor_bank(
        tokens,
        validation_range,
        sequence_length=sequence_length,
        token_states=int(row.get("anchor_token_states", 8192)) if trace_level != "off" else 0,
        seed=int(row["anchor_seed"]),
    )
    recorder = ParameterTraceRecorder(model)
    window = WindowDynamicsAccumulator()
    anchor_reference: AnchorReference | None = None
    traces: list[dict[str, Any]] = []
    key_snapshots: list[str] = []
    model_snapshots: list[str] = []
    save_snapshots = bool(row.get("save_snapshots", False))
    snapshot_root = output_root / "snapshots" / str(row["row_id"])

    def record_point(*, tokens_seen: int, step: int, phase: str, train_loss: float | None, checkpoint_target: int | None) -> None:
        nonlocal anchor_reference
        if trace_level == "off":
            validation, probes = _validation_with_probes(
                model, tokens, validation_range, sequence_length=sequence_length,
                batch_size=batch_size, device=device, precision=precision,
            )
            behavior: dict[str, Any] | None = None
        else:
            behavior, anchor_reference = evaluate_anchor_behavior(
                model,
                anchor_bank,
                batch_size=int(row.get("anchor_batch_size", batch_size)),
                device=device,
                precision=precision,
                reference=anchor_reference,
                router_metric=str(row.get("router_metric", "dot")),
                router_temperature=float(row.get("router_temperature", 1.0)),
                top_k=int(row["top_k"]),
            )
            validation = float(behavior["anchor_loss"])
            probes = None
        point = recorder.record(
            model,
            arm=str(row["arm"]),
            seed=seed,
            tokens=tokens_seen,
            step=step,
            phase=phase,
            validation_loss=validation,
            train_loss=train_loss,
            probes=probes,
        )
        point["trace_schema_version"] = 2
        point["checkpoint_target_tokens"] = checkpoint_target
        point["behavior"] = behavior
        point["window_dynamics"] = window.summarize(reset=True)
        point["optimizer_state_l2_norm"] = optimizer_state_norms(model, optimizers)
        traces.append(point)

    record_point(tokens_seen=0, step=0, phase="dense_control" if not geometry else "fixed" if fixed else "pre_freeze", train_loss=None, checkpoint_target=0)
    if save_snapshots:
        if geometry:
            key_snapshots.append(_save_key_snapshot(model, snapshot_root / "keys_0.pt", {"tokens": 0, "phase": traces[-1]["phase"]}))
        model_snapshots.append(_save_model_snapshot(model, snapshot_root / "model_0.pt", row, {"tokens": 0}))

    tokens_seen = step = checkpoint_index = 0
    while checkpoint_index < len(checkpoints) and checkpoints[checkpoint_index] == 0:
        checkpoint_index += 1
    frozen = fixed
    freeze_tokens: int | None = 0 if fixed and geometry else None
    freeze_key_hash: str | None = initial_key_hash if fixed and geometry else None
    frozen_key_state = [parameter.detach().clone() for parameter in geometry] if fixed else None
    postfreeze_key_grad_observed = False
    geometry_steps = algebra_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    stride = max(int(row.get("window_sample_stride", 64)), 1)
    alternating_ratio = int(row.get("alternating_ratio") or 1)
    while tokens_seen < target_tokens:
        if geometry and not fixed and freeze_fraction < 1.0 and not frozen and tokens_seen >= freeze_target:
            _set_trainable(algebra, geometry, algebra_active=True, geometry_active=False)
            frozen = True
            freeze_tokens = tokens_seen
            freeze_key_hash = key_hash(model)
            frozen_key_state = [parameter.detach().clone() for parameter in geometry]
            record_point(tokens_seen=tokens_seen, step=step, phase="freeze_event", train_loss=None, checkpoint_target=None)
            model.train()
        next_tokens = min(target_tokens, tokens_seen + batch_size * sequence_length)
        checkpoint_due = checkpoint_index < len(checkpoints) and next_tokens >= checkpoints[checkpoint_index]
        geometry_phase = mode == "alternating_adamw" and not frozen and ((step + 1) % (alternating_ratio + 1) == 0)
        if mode == "alternating_adamw":
            _set_trainable(algebra, geometry, algebra_active=not geometry_phase, geometry_active=geometry_phase)
            optimizer = geometry_optimizer if geometry_phase else algebra_optimizer
        else:
            _set_trainable(algebra, geometry, algebra_active=True, geometry_active=bool(geometry and not frozen and not fixed))
            optimizer = joint_optimizer
        if optimizer is None:
            raise AssertionError("registered optimizer has no parameters")
        optimizer.zero_grad(set_to_none=True)
        starts = starts_schedule[step]
        inputs, targets = _sample_windows(tokens, starts, sequence_length)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer_sample = trace_level != "off" and ((step + 1) % stride == 0 or checkpoint_due)
        pre_step = recorder.pre_step_snapshot() if optimizer_sample else None
        with _autocast(device, precision):
            logits = training_model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite behavioral-atlas loss")
        loss.backward()
        raw_gradients = recorder.gradient_norms() if optimizer_sample else None
        nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], float(row.get("gradient_clip_norm", 1.0)))
        clipped_gradients = recorder.gradient_norms() if optimizer_sample else None
        if frozen and any(parameter.grad is not None for parameter in geometry):
            postfreeze_key_grad_observed = True
        optimizer.step()
        if optimizer_sample and pre_step is not None and raw_gradients is not None and clipped_gradients is not None:
            window.observe(
                raw_gradients=raw_gradients,
                clipped_gradients=clipped_gradients,
                updates=exact_step_update_metrics(model, pre_step),
            )
        geometry_steps += int(bool(geometry) and (geometry_phase or (mode != "alternating_adamw" and not frozen and not fixed)))
        algebra_steps += int(not geometry_phase)
        step += 1
        tokens_seen += inputs.numel()
        if checkpoint_due:
            phase = "dense_control" if not geometry else "fixed" if fixed else "post_freeze" if frozen else "pre_freeze"
            checkpoint_target = checkpoints[checkpoint_index]
            record_point(tokens_seen=tokens_seen, step=step, phase=phase, train_loss=float(loss.detach()), checkpoint_target=checkpoint_target)
            if save_snapshots and geometry:
                key_snapshots.append(_save_key_snapshot(model, snapshot_root / f"keys_{checkpoint_target}.pt", {"tokens": tokens_seen, "target_tokens": checkpoint_target, "phase": phase}))
            while checkpoint_index < len(checkpoints) and tokens_seen >= checkpoints[checkpoint_index]:
                checkpoint_index += 1
            model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    test_loss = _validation_language(
        model, tokens, test_range, sequence_length=sequence_length, batch_size=batch_size,
        device=device, precision=precision,
    )
    final_key_hash = key_hash(model)
    postfreeze_drift = 0.0
    if frozen_key_state is not None:
        delta2 = sum(float((parameter.detach().float() - frozen_value.float()).square().sum()) for parameter, frozen_value in zip(geometry, frozen_key_state))
        base2 = sum(float(frozen_value.float().square().sum()) for frozen_value in frozen_key_state)
        postfreeze_drift = math.sqrt(delta2) / max(math.sqrt(base2), 1e-30)
    symmetry_bank = anchor_bank
    if geometry and symmetry_bank.inputs.numel() == 0:
        symmetry_bank = build_anchor_bank(tokens, validation_range, sequence_length=sequence_length, token_states=256, seed=int(row["anchor_seed"]))
    symmetry = matched_key_expert_permutation_check(
        model,
        symmetry_bank.inputs,
        device=device,
        precision=precision,
        seed=seed + 99,
    )
    restart_state_hash_match: bool | None = None
    if save_snapshots:
        final_path = snapshot_root / f"model_{target_tokens}.pt"
        model_snapshots.append(_save_model_snapshot(model, final_path, row, {"tokens": tokens_seen, "phase": traces[-1]["phase"]}))
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        restored = build_behavioral_atlas_model(row)
        restored.load_state_dict(payload["model"])
        restart_state_hash_match = state_hash(restored) == state_hash(model)
    trace_root = output_root / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_path = trace_root / f"{row['row_id']}.jsonl"
    trace_path.write_text("".join(json.dumps(point, sort_keys=True) + "\n" for point in traces), encoding="utf-8")
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    final_validation = float(traces[-1]["validation_loss"])
    result = {
        **row,
        "status": "pass",
        "failure_category": None,
        "dataset": dataset,
        "sample_order_sha256": sample_order_hash,
        "anchor_sha256": anchor_bank.sha256,
        "anchor_token_states_resolved": anchor_bank.token_states,
        "initial_state_hash": initial_state_hash,
        "initial_key_hash": initial_key_hash,
        "final_key_hash": final_key_hash,
        "total_parameters": total_parameters,
        "active_parameters_per_token": int(model.active_parameters_per_token),
        "target_tokens_resolved": target_tokens,
        "tokens": tokens_seen,
        "steps": step,
        "wall_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "validation_loss": final_validation,
        "test_loss": test_loss,
        "test_perplexity": math.exp(min(test_loss, 20)),
        "geometry_steps": geometry_steps,
        "algebra_steps": algebra_steps,
        "freeze_tokens": freeze_tokens,
        "freeze_fraction_observed": freeze_tokens / tokens_seen if freeze_tokens is not None and tokens_seen else None,
        "freeze_key_hash": freeze_key_hash,
        "postfreeze_key_hash_unchanged": bool(freeze_key_hash == final_key_hash) if freeze_key_hash else None,
        "postfreeze_relative_l2_drift": postfreeze_drift,
        "postfreeze_key_grad_observed": postfreeze_key_grad_observed,
        "optimizer_provenance": optimizer_provenance,
        "matched_key_expert_permutation": symmetry,
        "restart_state_hash_match": restart_state_hash_match,
        "trace_path": str(trace_path),
        "trace_points": len(traces),
        "traces": traces,
        "key_snapshots": key_snapshots,
        "model_snapshots": model_snapshots,
        "peak_vram_bytes": peak_vram,
        "execution": execution,
    }
    row_root = output_root / "rows" / "behavioral_atlas_v2"
    row_root.mkdir(parents=True, exist_ok=True)
    destination = row_root / f"{row['row_id']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


def run_manifest_row(manifest: str | Path, index: int, *, device: str, output_root: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if index < 0 or index >= len(rows):
        raise IndexError(f"manifest index {index} outside 0..{len(rows)-1}")
    return run_behavioral_atlas_row(rows[index], device=device, output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_manifest_row(args.manifest, args.index, device=args.device, output_root=args.output_root)
    print(json.dumps({
        "row_id": result["row_id"],
        "status": result["status"],
        "tokens": result["tokens"],
        "test_loss": result["test_loss"],
        "trace_points": result["trace_points"],
        "symmetry_passed": result["matched_key_expert_permutation"]["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
