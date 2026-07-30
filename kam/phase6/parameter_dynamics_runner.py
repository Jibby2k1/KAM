"""L4-oriented runner for the matched Phase 6.1 parameter-dynamics study."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kam.memory import SparseMemoryConfig, SparseSeparableMemory
from kam.phase6.overnight_runner import _autocast, _language_corpus, _sample_windows, _validation_language
from kam.phase6.parameter_trace import ParameterTraceRecorder, key_hash, state_hash
from kam.transformer.config import TransformerConfig
from kam.transformer.decoder import ModernDecoder


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def configure_l4_execution(device: torch.device) -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        properties = torch.cuda.get_device_properties(device)
        return {"device_name": properties.name, "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()), "fused_adamw": True, "tf32": True}
    return {"device_name": "cpu", "compute_capability": None, "bf16_supported": False, "fused_adamw": False, "tf32": False}


def build_parameter_dynamics_model(row: dict[str, Any]) -> ModernDecoder:
    config = TransformerConfig(d_model=int(row["d_model"]), n_heads=int(row["n_heads"]), n_layers=int(row["n_layers"]),
                               d_ff=int(row["d_ff"]), vocab_size=int(row.get("vocab_size", 256)),
                               max_seq_len=int(row["sequence_length"]))
    fixed = str(row["arm"]) == "fixed_keys"
    layers = [SparseSeparableMemory(SparseMemoryConfig(d_model=config.d_model, num_supports=int(row["num_supports"]),
                                                       top_k=int(row["top_k"]), expert_mode="low_rank",
                                                       expert_rank=int(row["expert_rank"]),
                                                       geometry_mode="fixed_random" if fixed else "learned_full"),
                                    seed=int(row["seed"]) + index)
              for index in range(config.n_layers)]
    return ModernDecoder(config, memory_layers=layers)


def _optimizer(parameters, *, lr: float, weight_decay: float, device: torch.device):
    values = list(parameters)
    if not values: return None
    options = {"lr": lr, "weight_decay": weight_decay}
    if device.type == "cuda": options["fused"] = True
    return torch.optim.AdamW(values, **options)


def _joint_optimizer(algebra, geometry, device: torch.device):
    groups = [{"params": list(algebra), "lr": 3e-4, "weight_decay": 0.1}]
    if geometry: groups.append({"params": list(geometry), "lr": 3e-5, "weight_decay": 0.0})
    options: dict[str, Any] = {}
    if device.type == "cuda": options["fused"] = True
    return torch.optim.AdamW(groups, **options)


def _validation_with_probes(model: nn.Module, tokens: Tensor, validation_range: tuple[int, int], *, sequence_length: int,
                            batch_size: int, device: torch.device, precision: str) -> tuple[float, list[Tensor]]:
    captured: list[Tensor | None] = [None] * len(getattr(model, "memory_layers", [])); hooks = []
    for index, layer in enumerate(getattr(model, "memory_layers", [])):
        def capture(_module, inputs, layer_index=index):
            if captured[layer_index] is None:
                captured[layer_index] = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:256].clone()
        hooks.append(layer.register_forward_pre_hook(capture))
    generator = torch.Generator().manual_seed(602214); low, high = validation_range
    starts = torch.randint(low, high - sequence_length - 1, (min(batch_size, 16),), generator=generator)
    inputs, targets = _sample_windows(tokens, starts, sequence_length)
    model.eval()
    with torch.inference_mode(), _autocast(device, precision):
        logits = model(inputs.to(device, non_blocking=True))
        loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.to(device, non_blocking=True).flatten())
    for hook in hooks: hook.remove()
    return float(loss), [value if value is not None else torch.empty(0, model.config.d_model, device=device) for value in captured]


def _save_key_snapshot(model: nn.Module, path: Path, metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {f"layer{index}": layer.keys.detach().to(device="cpu", dtype=torch.float16)
            for index, layer in enumerate(getattr(model, "memory_layers", []))}
    torch.save({"keys": keys, "metadata": metadata}, path)
    return str(path)


def _save_model_snapshot(model: nn.Module, path: Path, row: dict[str, Any], metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "row": row, "metadata": metadata}, path)
    return str(path)


def _set_trainable(algebra: list[nn.Parameter], geometry: list[nn.Parameter], *, algebra_active: bool, geometry_active: bool) -> None:
    for parameter in algebra: parameter.requires_grad_(algebra_active)
    for parameter in geometry:
        parameter.requires_grad_(geometry_active)
        if not geometry_active: parameter.grad = None


def run_parameter_dynamics_row(row: dict[str, Any], *, device: str | torch.device, output_root: str | Path) -> dict[str, Any]:
    device = torch.device(device); output_root = Path(output_root); seed = int(row["seed"])
    _seed_everything(seed); execution = configure_l4_execution(device)
    tokens, dataset = _language_corpus(row); train_end = int(dataset["train_range"][1])
    validation_range = tuple(int(value) for value in dataset["validation_range"])
    test_range = tuple(int(value) for value in dataset["test_range"])
    sequence_length, batch_size = int(row["sequence_length"]), int(row["batch_size"])
    model = build_parameter_dynamics_model(row).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    target_parameter_budget = int(row.get("target_parameter_budget", total_parameters))
    tolerance = float(row.get("parameter_tolerance_fraction", 0.01))
    if abs(total_parameters - target_parameter_budget) / max(target_parameter_budget, 1) > tolerance:
        raise AssertionError("locked model is outside the registered parameter tolerance")
    initial_state_hash, initial_key_hash = state_hash(model), key_hash(model)
    geometry = [layer.keys for layer in model.memory_layers]; geometry_ids = {id(parameter) for parameter in geometry}
    algebra = [parameter for parameter in model.parameters() if id(parameter) not in geometry_ids]
    fixed = str(row["arm"]) == "fixed_keys"; mode = str(row["optimization"])
    if fixed: _set_trainable(algebra, geometry, algebra_active=True, geometry_active=False)
    joint_optimizer = _joint_optimizer(algebra, [] if fixed else geometry, device) if mode != "alternating_8_1" else None
    algebra_optimizer = _optimizer(algebra, lr=3e-4, weight_decay=0.1, device=device) if mode == "alternating_8_1" else None
    geometry_optimizer = _optimizer(geometry, lr=3e-5, weight_decay=0.0, device=device) if mode == "alternating_8_1" else None
    smoke_tokens = os.environ.get("PHASE6_PARAMETER_DYNAMICS_SMOKE_TOKENS")
    target_tokens = min(int(row["target_tokens"]), int(smoke_tokens)) if smoke_tokens else int(row["target_tokens"])
    checkpoints = sorted({int(value) for value in row["validation_token_checkpoints"] if 0 <= int(value) <= target_tokens} | {0, target_tokens})
    freeze_fraction = float(row["freeze_fraction"]); freeze_target = int(freeze_fraction * target_tokens)
    recorder = ParameterTraceRecorder(model); traces: list[dict[str, Any]] = []; key_snapshots: list[str] = []; model_snapshots: list[str] = []
    precision = str(row.get("precision", "bf16")); save_snapshots = bool(row.get("save_snapshots", True))
    initial_validation, probes = _validation_with_probes(model, tokens, validation_range, sequence_length=sequence_length,
                                                         batch_size=batch_size, device=device, precision=precision)
    traces.append(recorder.record(model, arm=str(row["arm"]), seed=seed, tokens=0, step=0, phase="fixed" if fixed else "pre_freeze",
                                  validation_loss=initial_validation, train_loss=None, probes=probes))
    traces[-1]["checkpoint_target_tokens"] = 0
    snapshot_root = output_root / "snapshots" / str(row["row_id"])
    if save_snapshots:
        key_snapshots.append(_save_key_snapshot(model, snapshot_root / "keys_0.pt", {"tokens": 0, "phase": traces[-1]["phase"]}))
        model_snapshots.append(_save_model_snapshot(model, snapshot_root / "model_0.pt", row, {"tokens": 0}))
    generator = torch.Generator().manual_seed(int(row["data_seed"])); tokens_seen = step = checkpoint_index = 0
    while checkpoint_index < len(checkpoints) and checkpoints[checkpoint_index] == 0: checkpoint_index += 1
    frozen = fixed; freeze_tokens: int | None = 0 if fixed else None; freeze_key_hash: str | None = initial_key_hash if fixed else None
    frozen_key_state = [parameter.detach().clone() for parameter in geometry] if fixed else None
    postfreeze_key_grad_observed = False; geometry_steps = algebra_steps = 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); model.train()
    while tokens_seen < target_tokens:
        if not fixed and freeze_fraction < 1.0 and not frozen and tokens_seen >= freeze_target:
            _set_trainable(algebra, geometry, algebra_active=True, geometry_active=False); frozen = True; freeze_tokens = tokens_seen
            freeze_key_hash = key_hash(model)
            frozen_key_state = [parameter.detach().clone() for parameter in geometry]
            validation, probes = _validation_with_probes(model, tokens, validation_range, sequence_length=sequence_length,
                                                         batch_size=batch_size, device=device, precision=precision)
            traces.append(recorder.record(model, arm=str(row["arm"]), seed=seed, tokens=tokens_seen, step=step, phase="freeze_event",
                                          validation_loss=validation, train_loss=None, probes=probes))
            traces[-1]["checkpoint_target_tokens"] = None
            model.train()
        next_tokens = min(target_tokens, tokens_seen + batch_size * sequence_length)
        due = checkpoint_index < len(checkpoints) and next_tokens >= checkpoints[checkpoint_index]
        geometry_phase = mode == "alternating_8_1" and not frozen and (step + 1) % 9 == 0
        if mode == "alternating_8_1":
            _set_trainable(algebra, geometry, algebra_active=not geometry_phase, geometry_active=geometry_phase)
            optimizer = geometry_optimizer if geometry_phase else algebra_optimizer
        else:
            _set_trainable(algebra, geometry, algebra_active=True, geometry_active=not frozen and not fixed)
            optimizer = joint_optimizer
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        starts = torch.randint(0, train_end - sequence_length - 1, (batch_size,), generator=generator)
        inputs, targets = _sample_windows(tokens, starts, sequence_length)
        inputs = inputs.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
        with _autocast(device, precision):
            logits = model(inputs); loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite parameter-dynamics loss")
        loss.backward(); raw_gradients = recorder.gradient_norms() if due else None
        pre_step = recorder.pre_step_snapshot() if due else None
        nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        clipped_gradients = recorder.gradient_norms() if due else None
        if frozen and any(parameter.grad is not None for parameter in geometry): postfreeze_key_grad_observed = True
        optimizer.step(); geometry_steps += int(geometry_phase or (mode != "alternating_8_1" and not frozen and not fixed)); algebra_steps += int(not geometry_phase)
        step += 1; tokens_seen += inputs.numel()
        if due:
            validation, probes = _validation_with_probes(model, tokens, validation_range, sequence_length=sequence_length,
                                                         batch_size=batch_size, device=device, precision=precision)
            phase = "fixed" if fixed else "post_freeze" if frozen else "pre_freeze"
            traces.append(recorder.record(model, arm=str(row["arm"]), seed=seed, tokens=tokens_seen, step=step, phase=phase,
                                          validation_loss=validation, train_loss=float(loss.detach()), probes=probes,
                                          raw_gradients=raw_gradients, clipped_gradients=clipped_gradients, pre_step=pre_step))
            checkpoint_target = checkpoints[checkpoint_index]
            traces[-1]["checkpoint_target_tokens"] = checkpoint_target
            if save_snapshots:
                key_snapshots.append(_save_key_snapshot(model, snapshot_root / f"keys_{checkpoint_target}.pt",
                                                        {"tokens": tokens_seen, "target_tokens": checkpoint_target, "phase": phase}))
                if checkpoint_target in {40_000_000, target_tokens}:
                    model_snapshots.append(_save_model_snapshot(model, snapshot_root / f"model_{checkpoint_target}.pt", row,
                                                                {"tokens": tokens_seen, "phase": phase}))
            while checkpoint_index < len(checkpoints) and tokens_seen >= checkpoints[checkpoint_index]: checkpoint_index += 1
            model.train()
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_validation = traces[-1]["validation_loss"]
    test_loss = _validation_language(model, tokens, test_range, sequence_length=sequence_length, batch_size=batch_size,
                                     device=device, precision=precision)
    final_key_hash = key_hash(model); postfreeze_drift = 0.0
    if frozen_key_state is not None:
        delta2 = sum(float((parameter.detach().float() - frozen.float()).square().sum()) for parameter, frozen in zip(geometry, frozen_key_state))
        base2 = sum(float(frozen.float().square().sum()) for frozen in frozen_key_state)
        postfreeze_drift = math.sqrt(delta2) / max(math.sqrt(base2), 1e-30)
    trace_root = output_root / "traces"; trace_root.mkdir(parents=True, exist_ok=True)
    trace_path = trace_root / f"{row['row_id']}.jsonl"
    trace_path.write_text("".join(json.dumps(point, sort_keys=True) + "\n" for point in traces), encoding="utf-8")
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    result = {**row, "status": "pass", "failure_category": None, "dataset": dataset, "initial_state_hash": initial_state_hash,
              "initial_key_hash": initial_key_hash, "final_key_hash": final_key_hash, "total_parameters": total_parameters,
              "active_parameters_per_token": int(model.active_parameters_per_token), "target_tokens_resolved": target_tokens,
              "tokens": tokens_seen, "steps": step, "wall_seconds": elapsed, "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
              "validation_loss": final_validation, "test_loss": test_loss, "test_perplexity": math.exp(min(test_loss, 20)),
              "geometry_steps": geometry_steps, "algebra_steps": algebra_steps, "freeze_tokens": freeze_tokens,
              "freeze_fraction_observed": freeze_tokens / tokens_seen if freeze_tokens is not None and tokens_seen else None,
              "freeze_key_hash": freeze_key_hash, "postfreeze_key_hash_unchanged": bool(freeze_key_hash == final_key_hash) if freeze_key_hash else None,
              "postfreeze_relative_l2_drift": postfreeze_drift, "postfreeze_key_grad_observed": postfreeze_key_grad_observed,
              "trace_path": str(trace_path), "trace_points": len(traces), "traces": traces,
              "key_snapshots": key_snapshots, "model_snapshots": model_snapshots, "peak_vram_bytes": peak_vram,
              "execution": execution}
    row_root = output_root / "rows" / "parameter_dynamics_v1"; row_root.mkdir(parents=True, exist_ok=True)
    destination = row_root / f"{row['row_id']}.json"; temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(destination)
    return result


def run_manifest_row(manifest: str | Path, index: int, *, device: str, output_root: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if index < 0 or index >= len(rows): raise IndexError(f"manifest index {index} outside 0..{len(rows)-1}")
    return run_parameter_dynamics_row(rows[index], device=device, output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True); parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); result = run_manifest_row(args.manifest, args.index, device=args.device, output_root=args.output_root)
    print(json.dumps({"row_id": result["row_id"], "status": result["status"], "tokens": result["tokens"],
                      "test_loss": result["test_loss"], "trace_points": result["trace_points"]}, indent=2))


if __name__ == "__main__": main()
