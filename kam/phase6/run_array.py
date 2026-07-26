from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from kam.data.phase6 import (
    DynamicsConfig,
    controlled_prototype,
    controlled_symbolic_regimes,
    language_batches,
    load_text_tokens,
    lorenz63,
    mqar,
    scheduled_stream,
    switching_mackey_glass,
    switching_narma,
    rossler,
)
from kam.memory import EpisodicMemory, MemoryTokenLayer, SparseMemoryConfig, SparseSeparableMemory
from kam.optimization import AlternatingSchedule, NLMSReadout, RLSReadout, dictionary_update, ridge_solve
from kam.phase6.diagnostics import measure_forward_backward, measure_forward, resource_accounting
from kam.phase6.plots import plot_learning_curves, plot_prediction_true_error
from kam.transformer import build_baseline

from .run_stage0 import run_row as run_stage0_row, seed_everything


def _device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _measure_decoder(model: torch.nn.Module, *, vocab_size: int, sequence_length: int, device: torch.device) -> dict[str, float]:
    model.train()
    tokens = torch.randint(vocab_size, (1, sequence_length), device=device)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits = model(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_ms = (time.perf_counter() - start) * 1000.0
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    logits.square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_ms = (time.perf_counter() - start) * 1000.0
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    tokens_seen = sequence_length
    return {
        "measured_forward_ms": forward_ms,
        "measured_backward_ms": backward_ms,
        "tokens_or_samples_per_second": tokens_seen / max((forward_ms + backward_ms) / 1000.0, 1e-12),
        "peak_vram": peak,
    }


def _as_model_input(features: Tensor, d_model: int) -> Tensor:
    features = features.float()
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.shape[-1] < d_model:
        features = F.pad(features, (0, d_model - features.shape[-1]))
    return features[..., :d_model]


class _ZeroMemory(nn.Module):
    def forward(self, query: Tensor, return_diagnostics: bool = False):
        update = torch.zeros_like(query)
        return (update, {}) if return_diagnostics else update


class _MechanismModel(nn.Module):
    def __init__(self, d_model: int, architecture: str, memory: nn.Module) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)) if architecture == "T-WIDE" else nn.Linear(d_model, d_model)
        self.memory = memory

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        base = self.backbone(query)
        result = self.memory(query, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            update, diagnostics = result
            return base + update, diagnostics
        return base + result


def _dynamics_task(task: str, seed: int) -> tuple[Tensor, Tensor]:
    if task in {"prototype", "prototype_schedule"}:
        return controlled_prototype(length=96, regime=seed % 2, seed=seed)
    if "mackey" in task:
        features, targets, _ = switching_mackey_glass(length=96, config=DynamicsConfig(length=96, seed=seed))
        return features, targets
    if "narma" in task:
        features, targets, _ = switching_narma(length=96, config=DynamicsConfig(length=96, seed=seed))
        return features, targets
    if task == "lorenz":
        return lorenz63(length=96, seed=seed)
    if task == "rossler":
        return rossler(length=96)
    raise ValueError(f"unsupported dynamics task: {task}")


def _mechanism_task(task: str, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return continuous features, targets, and a valid-loss mask for Stage 1."""
    if task == "mqar":
        inputs, targets = mqar(batch=1, pairs=4, sequence_length=96, vocab_size=32, seed=seed)
        features = inputs.float().unsqueeze(-1) / 31.0
        mask = targets.ne(-100).unsqueeze(-1)
        clean_targets = targets.clamp_min(0).float().unsqueeze(-1) / 31.0
        return features[0], clean_targets[0], mask[0]
    features, targets = _dynamics_task(task, seed)
    return features, targets, torch.ones_like(targets, dtype=torch.bool)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    expanded_mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
    expanded_mask = expanded_mask.expand_as(prediction)
    values = (prediction - target).square()[expanded_mask]
    return values.mean() if values.numel() else (prediction - target).square().mean()


def _last_linear(module: _MechanismModel) -> nn.Linear:
    for child in reversed(list(module.backbone.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise TypeError("mechanism backbone must expose a final linear readout")


def _ridge_design(module: _MechanismModel, query: Tensor, readout: nn.Linear) -> Tensor:
    """Return the features consumed by the final linear readout.

    T-WIDE has a hidden expansion before its final projection, so fitting a
    ridge readout on the backbone output would produce a [d_model, d_model]
    solution for a layer that expects [d_model, 2*d_model].
    """
    if isinstance(module.backbone, nn.Sequential):
        hidden = query
        for layer in module.backbone:
            if layer is readout:
                return hidden
            hidden = layer(hidden)
    return query


def _ridge_readout_update(module: _MechanismModel, query: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    """Solve and install the final linear readout for algebra-solve rows."""
    with torch.no_grad():
        readout = _last_linear(module)
        hidden = _ridge_design(module, query, readout)
        features = hidden.reshape(-1, hidden.shape[-1])
        targets = target.reshape(-1, target.shape[-1])
        valid = mask.reshape(-1).bool()
        features = features[valid]
        targets = targets[valid]
        augmented = torch.cat((features, torch.ones(features.shape[0], 1, device=features.device)), dim=-1)
        solved = ridge_solve(augmented, targets, regularization=1e-4)
        readout.weight.copy_(solved.solution[:-1].T)
        if readout.bias is not None:
            readout.bias.copy_(solved.solution[-1])
    return {"solver_condition_number": float(solved.condition_number), "solver_residual_norm": float(solved.residual_norm)}


def _mechanism_train(
    module: _MechanismModel,
    query: Tensor,
    target: Tensor,
    mask: Tensor,
    row: dict[str, Any],
    loss_history: list[float],
) -> dict[str, float | str]:
    """Apply the declared joint, alternating, solve, or dictionary update mode."""
    label = str(row.get("optimizer", "joint_sgd"))
    steps = max(1, int(4 * float(row.get("fidelity", 1.0))))
    all_parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    geometry_parameters = [module.memory.keys] if isinstance(getattr(module, "memory", None), SparseSeparableMemory) and module.memory.keys.requires_grad else []
    geometry_ids = {id(parameter) for parameter in geometry_parameters}
    algebra_parameters = [parameter for parameter in all_parameters if id(parameter) not in geometry_ids]
    diagnostics: dict[str, float | str] = {"optimizer_mode": label}

    def current_loss() -> Tensor:
        return _masked_mse(module(query), target, mask)

    if label == "ridge_resolve" or label.startswith("variable_projection"):
        for _ in range(steps):
            diagnostics.update(_ridge_readout_update(module, query, target, mask))
            if label.startswith("variable_projection") and geometry_parameters:
                geometry_optimizer = torch.optim.Adam(geometry_parameters, lr=0.005)
                module.train()
                geometry_optimizer.zero_grad(set_to_none=True)
                projected_loss = current_loss()
                projected_loss.backward()
                geometry_optimizer.step()
            loss_history.append(float(current_loss().detach()))
        diagnostics["optimizer_steps"] = float(steps)
        return diagnostics

    if label == "dictionary_update" and isinstance(getattr(module, "memory", None), SparseSeparableMemory):
        with torch.no_grad():
            updated, dictionary_diagnostics = dictionary_update(query.detach().reshape(-1, query.shape[-1]), module.memory.keys.detach())
            geometry_result = module.memory.update_geometry(updated, trust_radius=float("inf"))
        diagnostics.update({f"dictionary_{key}": float(value) for key, value in dictionary_diagnostics.items()})
        diagnostics["dictionary_geometry_accepted"] = float(bool(geometry_result.get("accepted", False)))

    if label.startswith("alternating_"):
        parts = label.split("_")
        schedule = AlternatingSchedule.from_label(f"alternating_{parts[1]}:{parts[2]}")
        algebra_optimizer = torch.optim.Adam(algebra_parameters or all_parameters, lr=0.01)
        geometry_optimizer = torch.optim.Adam(geometry_parameters, lr=0.005) if geometry_parameters else None
        geometry_steps = 0
        for step in range(steps):
            phase = schedule.phase(step)
            optimizer = algebra_optimizer if phase == "algebra" or geometry_optimizer is None else geometry_optimizer
            if phase == "geometry" and geometry_optimizer is not None:
                geometry_steps += 1
            module.train()
            optimizer.zero_grad(set_to_none=True)
            loss = current_loss()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach()))
        if geometry_optimizer is not None and geometry_steps == 0:
            module.train()
            geometry_optimizer.zero_grad(set_to_none=True)
            geometry_loss = current_loss()
            geometry_loss.backward()
            geometry_optimizer.step()
            loss_history.append(float(geometry_loss.detach()))
            geometry_steps = 1
        diagnostics["alternating_geometry_steps"] = float(geometry_steps)
        diagnostics["alternating_declared_algebra_steps"] = float(schedule.algebra_steps)
        diagnostics["alternating_declared_geometry_steps"] = float(schedule.geometry_steps)
        diagnostics["optimizer_steps"] = float(steps + (1 if geometry_optimizer is not None and steps > 0 and geometry_steps > sum(schedule.phase(step) == "geometry" for step in range(steps)) else 0))
        return diagnostics

    optimizer = torch.optim.Adam(all_parameters, lr=0.01) if all_parameters else None
    for _ in range(steps):
        if optimizer is None:
            break
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss = current_loss()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    diagnostics["optimizer_steps"] = float(steps)
    return diagnostics


def _transformer_task(task: str, seed: int) -> tuple[Tensor, Tensor, int]:
    """Create the declared token fixture for transformer comparison rows."""
    if task == "mqar":
        return (*mqar(batch=2, pairs=3, sequence_length=32, vocab_size=64, seed=seed), 64)
    if task == "controlled_symbolic_regimes":
        inputs, targets, _ = controlled_symbolic_regimes(batch=2, length=32, modulus=7, seed=seed)
        return inputs, targets, 7
    if task == "small_language":
        tokens, vocabulary = load_text_tokens()
        inputs, targets = language_batches(tokens, batch_size=2, sequence_length=32, seed=seed)
        return inputs, targets, len(vocabulary)
    if task == "prototype":
        features, targets = controlled_prototype(length=64, regime=seed % 2, seed=seed)
        encode = lambda values: (((values[:, 0].clamp(-1.0, 1.0) + 1.0) * 31.5).round().long()).unsqueeze(0)
        return encode(features), encode(targets), 64
    if task in {"switching_mackey_glass", "switching_narma"}:
        features, targets = _dynamics_task(task, seed)
        features, targets = features[:64], targets[:64]

        def encode(values: Tensor) -> Tensor:
            # Keep the continuous dynamics fixtures deterministic and bounded
            # when they enter the causal decoder token interface.
            clipped = values.float().clamp(-2.0, 2.0)
            return (((clipped + 2.0) * 15.75).round().long()).reshape(1, -1)

        return encode(features), encode(targets), 64
    raise ValueError(f"unsupported transformer task: {task}")


def _online_task(task: str, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    """Create the A→B→A→C→A stream required by online rows."""
    if "symbolic" in task:
        def make_stream(offset: int) -> tuple[Tensor, Tensor]:
            values, next_values, _ = controlled_symbolic_regimes(batch=1, length=48, seed=seed + offset)
            return values[0].float().unsqueeze(-1), next_values[0].float().unsqueeze(-1)
    else:
        base = "prototype" if "prototype" in task else "switching_mackey_glass" if "mackey" in task else "switching_narma"

        def make_stream(offset: int) -> tuple[Tensor, Tensor]:
            return _dynamics_task(base, seed + offset)

    streams = {name: make_stream(index * 17) for index, name in enumerate(("A", "B", "C"))}
    return scheduled_stream(streams)


def _run_mechanism(row: dict[str, Any], device: torch.device, run_root: Path | None) -> dict[str, Any]:
    features, targets, loss_mask = _mechanism_task(str(row.get("task", "prototype")), int(row["seed"]))
    d_model = int(row.get("d_model", 32))
    query_cpu = _as_model_input(features, d_model)
    query = query_cpu.to(device)
    geometry_mode = str(row.get("geometry", "learned_full"))
    architecture = str(row.get("architecture", "T-KAM-L"))
    if architecture in {"T0", "T-WIDE"}:
        memory: nn.Module = _ZeroMemory()
    elif architecture == "T-MEMTOK":
        memory = MemoryTokenLayer(d_model, num_tokens=min(int(row.get("num_supports", 32)), 128), top_k=min(int(row.get("top_k", 4)), 8))
    else:
        memory = SparseSeparableMemory(
            SparseMemoryConfig(
                d_model=d_model,
                num_supports=min(int(row.get("num_supports", 32)), 512),
                top_k=min(int(row.get("top_k", 4)), int(row.get("num_supports", 32))),
                expert_mode={"vector": "vector", "low_rank_affine_expert": "low_rank", "routes_only": "routes_only"}.get(str(row.get("expert", "vector")), "vector"),
                geometry_mode=geometry_mode,
            ),
            seed=int(row["seed"]),
            key_data=query_cpu[0].detach() if geometry_mode in {"fixed_data_sample", "fixed_kmeans", "fixed_farthest_point"} else None,
        )
    module = _MechanismModel(d_model, architecture, memory).to(device)
    target = _as_model_input(targets, d_model).to(device)
    module.eval()
    with torch.no_grad():
        initial_output = module(query)
        initial_loss = _masked_mse(initial_output, target, loss_mask.to(device))
    loss_history = [float(initial_loss.detach())]
    steps = max(1, int(4 * float(row.get("fidelity", 1.0))))
    optimizer_diagnostics = _mechanism_train(module, query, target, loss_mask.to(device), row, loss_history)
    module.eval()
    output, routing = module(query, return_diagnostics=True)
    loss = _masked_mse(output, target, loss_mask.to(device))
    memory_slots = int(getattr(getattr(module, "memory", None), "config", SparseMemoryConfig(num_supports=0)).num_supports)
    top_k = int(getattr(getattr(module, "memory", None), "config", SparseMemoryConfig(top_k=0)).top_k)
    metrics = {"initial_loss": float(initial_loss.detach()), "loss": float(loss.detach()), "loss_history": loss_history, "task_mask_fraction": float(loss_mask.float().mean()), "training_steps": float(optimizer_diagnostics.get("optimizer_steps", steps)), **optimizer_diagnostics, **routing, **resource_accounting(module, tokens=query.shape[1], sequence_length=query.shape[1], memory_slots=memory_slots, top_k=top_k)}
    metrics.update(measure_forward(module, batch_size=query.shape[0], sequence_length=query.shape[1], d_model=d_model, device=str(device)))
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
        metrics["prediction_error_figure"] = plot_prediction_true_error(targets[:, 0], output.detach().cpu()[0, :, 0], run_root / "prediction_true_error.png")
    return metrics


def _run_transformer(row: dict[str, Any], device: torch.device, run_root: Path | None) -> dict[str, Any]:
    architecture = str(row.get("architecture", "T0"))
    scale = str(row.get("scale", "tiny"))
    if scale not in {"tiny", "2M", "10M", "30M", "100M"}:
        scale = "tiny"
    inputs, targets, vocab_size = _transformer_task(str(row.get("task", "mqar")), int(row["seed"]))
    target_parameter_budget = int(row.get("target_parameter_budget", 0)) or None
    if scale not in {"2M", "10M", "30M", "100M"}:
        target_parameter_budget = None
    model, spec = build_baseline(architecture, scale=scale, vocab_size=vocab_size, max_seq_len=64, num_supports=min(int(row.get("num_supports", 32)), 128), top_k=min(int(row.get("top_k", 4)), 8), seed=int(row["seed"]), target_parameters=target_parameter_budget)
    model = model.to(device)
    inputs, targets = inputs.to(device), targets.to(device)
    criterion = lambda values: F.cross_entropy(values.reshape(-1, values.shape[-1]), targets.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        logits, diagnostics = model(inputs, return_diagnostics=True)
        initial_loss = criterion(logits)
    tokens_per_step = int(inputs.numel())
    declared_budget = int(row.get("token_budget", 0))
    if str(row.get("stage", "")) == "stage5_long_training" and declared_budget > 0:
        target_tokens = min(declared_budget, int(row.get("training_token_cap", declared_budget)))
    else:
        target_tokens = tokens_per_step * max(1, int(row.get("training_steps", 4)))
    training_steps = max(1, math.ceil(target_tokens / max(tokens_per_step, 1)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_history = [float(initial_loss.detach())]
    model.train()
    geometry_parameters = [parameter for name, parameter in model.named_parameters() if name.endswith(".keys") and parameter.requires_grad]
    geometry_ids = {id(parameter) for parameter in geometry_parameters}
    algebra_parameters = [parameter for parameter in model.parameters() if id(parameter) not in geometry_ids and parameter.requires_grad]
    alternating_optimizer = torch.optim.AdamW(algebra_parameters, lr=3e-4, weight_decay=1e-4) if architecture == "T-KAM-ALT" and algebra_parameters else None
    geometry_optimizer = torch.optim.AdamW(geometry_parameters, lr=1.5e-4, weight_decay=1e-4) if architecture == "T-KAM-ALT" and geometry_parameters else None
    if architecture == "T-KAM-VP":
        for parameter in geometry_parameters:
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(algebra_parameters, lr=3e-4, weight_decay=1e-4) if algebra_parameters else optimizer
    geometry_update_steps = 0
    algebra_update_steps = 0
    optimizer_mode = "joint_adamw"
    for step in range(training_steps):
        step_optimizer = optimizer
        if architecture == "T-KAM-ALT" and alternating_optimizer is not None:
            geometry_phase = geometry_optimizer is not None and (step == training_steps - 1 or (step + 1) % 9 == 0)
            step_optimizer = geometry_optimizer if geometry_phase else alternating_optimizer
            if geometry_phase:
                geometry_update_steps += 1
                optimizer_mode = "alternating_8_1"
            else:
                algebra_update_steps += 1
        elif architecture == "T-KAM-VP":
            optimizer_mode = "variable_projection_stopgrad"
            algebra_update_steps += 1
        else:
            algebra_update_steps += 1
        step_optimizer.zero_grad(set_to_none=True)
        train_logits = model(inputs)
        train_loss = criterion(train_logits)
        train_loss.backward()
        step_optimizer.step()
        loss_history.append(float(train_loss.detach()))
    model.eval()
    with torch.no_grad():
        logits, diagnostics = model(inputs, return_diagnostics=True)
        loss = criterion(logits)
    memory_layers = list(getattr(model, "memory_layers", []))
    effective_memory_slots = max(
        (
            int(getattr(layer, "num_experts", 0))
            or int(getattr(getattr(layer, "config", None), "num_supports", 0))
            or int(getattr(getattr(layer, "tokens", None), "shape", (0,))[0])
            or int(getattr(getattr(layer, "values", None), "shape", (0,))[0])
        )
        for layer in memory_layers
    ) if memory_layers else 0
    effective_top_k = max(
        (
            int(getattr(layer, "top_k", 0))
            or int(getattr(getattr(layer, "router", None), "top_k", 0))
            for layer in memory_layers
        ),
        default=int(row.get("top_k", 4)),
    )
    resource = resource_accounting(model, tokens=inputs.numel(), sequence_length=inputs.shape[1], memory_slots=effective_memory_slots, top_k=effective_top_k)
    total_parameters = float(resource["total_parameters"])
    resource.update({
        "target_parameter_budget": float(target_parameter_budget or 0),
        "parameter_match_error_fraction": float(abs(total_parameters - target_parameter_budget) / target_parameter_budget) if target_parameter_budget else 0.0,
    })
    metrics = {"initial_loss": float(initial_loss), "loss": float(loss), "perplexity": float(torch.exp(loss)), "loss_history": loss_history, "training_steps": float(training_steps), "training_tokens": float(training_steps * tokens_per_step), "declared_token_budget": float(declared_budget), "budget_completion_fraction": float(training_steps * tokens_per_step / declared_budget) if declared_budget else 1.0, "task": str(row.get("task")), "architecture": spec.name, "training_optimizer_mode": optimizer_mode, "geometry_update_steps": float(geometry_update_steps), "algebra_update_steps": float(algebra_update_steps), "declared_memory_slots": float(row.get("num_supports", 32)), "effective_memory_slots": float(effective_memory_slots), "effective_top_k": float(effective_top_k), **resource}
    for index, info in enumerate(diagnostics):
        for key, value in info.items():
            if isinstance(value, (float, int)):
                metrics[f"memory{index}_{key}"] = float(value)
    metrics.update(_measure_decoder(model, vocab_size=vocab_size, sequence_length=inputs.shape[1], device=device))
    return metrics


def _run_router_scaling(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from kam.memory.routers import ApproximateTopKRouter, ChunkedExactTopKRouter, ExactTopKRouter, ProductKeyRouter, recall_at_k, routing_diagnostics

    requested = int(row.get("slot", row.get("num_supports", 1000)))
    stage_mode = str(row.get("stage_mode", "profile"))
    profile_cap = int(row.get("profile_cap", 65536))
    benchmark = min(requested, profile_cap) if stage_mode == "profile" else requested
    d_model = int(row.get("d_model", 32))
    requested_precision = str(row.get("precision", "fp32")).lower()
    precision_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(requested_precision, torch.float32)
    if device.type != "cuda" and precision_dtype in {torch.bfloat16, torch.float16}:
        precision_dtype = torch.float32
    effective_precision = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[precision_dtype]
    query = torch.randn(16, d_model, device=device, dtype=precision_dtype)
    keys = torch.randn(benchmark, d_model, device=device, dtype=precision_dtype)
    reference_query = query.float()
    reference_keys = keys.float()
    kind = str(row.get("router", "exact"))
    factorized_codebook_size = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    route_start = time.perf_counter()
    if kind == "chunked":
        router = ChunkedExactTopKRouter(int(row.get("top_k", 4)), chunk_size=min(4096, benchmark))
        route = router(query, keys)
    elif kind == "approximate":
        router = ApproximateTopKRouter(int(row.get("top_k", 4)), candidate_size=min(4096, benchmark), seed=int(row["seed"]))
        route = router(query, keys)
    elif kind == "product_key":
        half = d_model // 2
        factorized_codebook_size = max(2, math.ceil(math.sqrt(benchmark)))
        product_keys = torch.randn(factorized_codebook_size, d_model, device=device, dtype=precision_dtype)
        router = ProductKeyRouter(int(row.get("top_k", 4)))
        codebook_a = product_keys[:, :half]
        codebook_b = product_keys[:, half:]
        route = router(query, codebook_a, codebook_b)
        materialized = torch.cat((codebook_a[:, None, :].expand(-1, codebook_b.shape[0], -1), codebook_b[None, :, :].expand(codebook_a.shape[0], -1, -1)), dim=-1).reshape(-1, d_model)
        reference_product_keys = product_keys.float()
        reference_materialized = torch.cat((reference_product_keys[:, None, :half].expand(-1, factorized_codebook_size, -1), reference_product_keys[None, :, half:].expand(factorized_codebook_size, -1, -1)), dim=-1).reshape(-1, d_model)
        reference = ExactTopKRouter(int(row.get("top_k", 4)))(reference_query, reference_materialized)
    else:
        router = ExactTopKRouter(int(row.get("top_k", 4)))
        route = router(query, keys)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    routing_ms = (time.perf_counter() - route_start) * 1000.0
    if kind in {"chunked", "approximate"}:
        reference = ExactTopKRouter(int(row.get("top_k", 4)))(reference_query, reference_keys)
    elif kind != "product_key":
        reference = ExactTopKRouter(int(row.get("top_k", 4)))(reference_query, reference_keys)
    precision_bytes = float(torch.tensor([], dtype=precision_dtype).element_size())
    benchmark_supports = int(route.num_supports)
    bank_storage_bytes = (
        float(2 * factorized_codebook_size * (d_model // 2) * precision_bytes)
        if kind == "product_key"
        else float(benchmark_supports * d_model * precision_bytes)
    )
    peak_vram = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    metrics = {"requested_supports": float(requested), "benchmark_supports": float(benchmark_supports), "router": kind, "stage_mode": stage_mode, "precision_requested": requested_precision, "precision_effective": effective_precision, "precision_bytes": precision_bytes, "bank_storage_bytes": bank_storage_bytes, "optimizer_state_bytes": 0.0, "routing_forward_ms": routing_ms, "routing_throughput_tokens_per_sec": float(query.shape[0] / max(routing_ms / 1000.0, 1e-12)), "peak_vram_bytes": peak_vram, "recall_at_k_against_exact": float(recall_at_k(route, reference)), **({"factorized_codebook_size": float(factorized_codebook_size)} if kind == "product_key" else {}), **route.diagnostics, **routing_diagnostics(route, route.num_supports)}
    return metrics


def _online_lift(features: Tensor) -> Tensor:
    """Use a deterministic four-channel representation for non-T0 online rows."""
    return torch.cat((features, features.square(), features.sin(), features.cos()), dim=-1)


def _run_online(row: dict[str, Any], device: torch.device, run_root: Path | None = None) -> dict[str, Any]:
    features, targets, schedule_labels = _online_task(str(row.get("task", "prototype_schedule")), int(row["seed"]))
    raw_features = _as_model_input(features, 1).to(device)
    targets = targets.to(device)
    architecture = str(row.get("architecture", "T0"))
    adapter = str(row.get("adapter", "none"))
    memory: SparseSeparableMemory | None = None
    routing: dict[str, float] = {}
    if architecture == "T0":
        representation = raw_features
    else:
        lifted = _online_lift(raw_features)
        if architecture == "T-WIDE":
            representation = lifted
        else:
            geometry = "fixed_random" if architecture == "T-KAM-F" else "learned_full"
            expert_mode = "vector" if architecture == "T-KAM-F" else "low_rank"
            memory = SparseSeparableMemory(
                SparseMemoryConfig(
                    d_model=lifted.shape[-1],
                    num_supports=min(int(row.get("num_supports", 32)), 128),
                    top_k=min(int(row.get("top_k", 4)), 8),
                    expert_mode=expert_mode,
                    expert_rank=1,
                    geometry_mode=geometry,
                    gate_init=1.0,
                ),
                seed=int(row["seed"]),
            ).to(device)
            with torch.no_grad():
                persistent, routing = memory(lifted, return_diagnostics=True)
            representation = lifted + persistent
    episodic_enabled = adapter == "episodic_insertion" or architecture in {"T-KAM-ONLINE", "T-KAM-DUAL"}
    episodic = EpisodicMemory(capacity=min(int(row.get("num_supports", 32)), 128), d_model=representation.shape[-1]).to(device) if episodic_enabled else None
    readout = RLSReadout(representation.shape[-1], output_dim=1) if adapter == "rls" else NLMSReadout(representation.shape[-1], output_dim=1) if adapter == "nlms" else None
    weights = torch.zeros(representation.shape[-1], 1, device=device)
    errors: list[float] = []
    geometry_updates = 0
    segment_length = max(1, representation.shape[1] // max(int(schedule_labels.unique().numel()), 1))
    for index in range(representation.shape[1]):
        if memory is not None and adapter == "slow_geometry" and index > 0 and index % segment_length == 0:
            with torch.no_grad():
                start = max(0, index - segment_length)
                candidate = memory.keys.detach() * 0.995 + _online_lift(raw_features)[0, start:index].mean(0).expand_as(memory.keys) * 0.005
                candidate = candidate / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                decision = memory.update_geometry(candidate)
                geometry_updates += int(bool(decision.get("accepted", False)))
                persistent, routing = memory(_online_lift(raw_features), return_diagnostics=True)
                representation = _online_lift(raw_features) + persistent
        x = representation[0, index]
        if episodic is not None and bool(episodic.valid.any()):
            x = x + episodic(x)
        y = targets[index].reshape(1)
        prediction = readout.predict(x) if readout is not None else x @ weights[:, 0]
        error = prediction.reshape(-1)[0] - y[0]
        errors.append(float(error.square().detach()))
        if readout is not None:
            readout.update(x, y)
        elif adapter in {"sgd", "value_only", "expert_only", "slow_geometry"}:
            # The online comparison must remain stable across heterogeneous
            # streams.  Raw SGD makes the integer-valued symbolic fixture and
            # lifted features explode, so use a bounded normalized update.
            if not torch.isfinite(x).all() or not torch.isfinite(error):
                raise FloatingPointError("non-finite online feature or error")
            feature_norm = x.detach().square().sum().clamp_min(1.0)
            bounded_error = error.detach().clamp(-10.0, 10.0)
            update = 0.05 * bounded_error * x.detach().unsqueeze(-1) / feature_norm
            weights.sub_(update.clamp(-0.5, 0.5)).clamp_(-10.0, 10.0)
        if episodic is not None:
            value = torch.zeros_like(x)
            value[0] = y[0]
            episodic.write(x.detach().reshape(1, -1), value.detach().reshape(1, -1))
    if not errors or not all(math.isfinite(value) for value in errors):
        raise FloatingPointError("non-finite online squared-error history")
    split = max(1, len(errors) // 3)
    variance = float(targets.detach().float().var().clamp_min(1e-12))
    metrics = {"architecture": architecture, "adapter": adapter, "stream_task": str(row.get("task")), "schedule_segments": float(schedule_labels.unique().numel()), "representation_dim": float(representation.shape[-1]), "memory_used": float(memory is not None), "episodic_active": float(episodic is not None), "geometry_update_count": float(geometry_updates), "global_nmse": float(np.mean(errors) / variance), "early_nmse": float(np.mean(errors[:split]) / variance), "late_nmse": float(np.mean(errors[-split:]) / variance), "reacquisition_time": float(next((index for index, value in enumerate(errors) if value < np.mean(errors[-split:])), len(errors))), "squared_error_history": errors, **routing}
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
        metrics["online_adaptation_figure"] = plot_learning_curves({"squared_error": errors}, run_root / "online_adaptation_curve.png")
    return metrics


def run_row(row: dict[str, Any], *, device: str = "auto", run_root: str | Path | None = None) -> dict[str, Any]:
    seed_everything(int(row["seed"]))
    result = dict(row)
    try:
        resolved = _device(device)
        root = Path(run_root) / str(row["row_id"]) if run_root else None
        stage = str(row.get("stage", ""))
        if stage == "stage0_validity":
            return run_stage0_row(row)
        if stage == "stage1_mechanism":
            metrics = _run_mechanism(row, resolved, root)
        elif stage == "stage3_router_scaling":
            metrics = _run_router_scaling(row, resolved)
        elif stage == "stage4_online_adaptation":
            metrics = _run_online(row, resolved, root)
        else:
            metrics = _run_transformer(row, resolved, root)
        result.update({"status": "pass", "metrics": metrics})
        return result
    except Exception as exc:  # noqa: BLE001 - preserve row-scoped scientific failure
        result.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        return result


def run_manifest(manifest_path: str | Path, output_path: str | Path, *, row_index: int | None = None, device: str = "auto", run_root: str | Path | None = None, resume: bool = False) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if row_index is not None:
        rows = [rows[row_index]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "pass":
            return {"rows": 1, "passed": 1, "failed": 0, "resumed": True}
    results = [run_row(row, device=device, run_root=run_root) for row in rows]
    if len(results) == 1:
        output.write_text(json.dumps(results[0], sort_keys=True) + "\n", encoding="utf-8")
    else:
        output.write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results), encoding="utf-8")
    return {"rows": len(results), "passed": sum(result.get("status") == "pass" for result in results), "failed": sum(result.get("status") != "pass" for result in results)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resumable Phase 6 manifest row")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-index", "--row-index", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_manifest(args.manifest, args.output, row_index=args.array_index, device=args.device, run_root=args.run_root, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
