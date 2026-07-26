from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def resource_accounting(
    module: nn.Module,
    *,
    tokens: int = 1,
    sequence_length: int = 1,
    memory_slots: int = 0,
    top_k: int = 0,
    dtype_bytes: int = 4,
    optimizer_state_multiplier: float = 2.0,
) -> dict[str, float]:
    """Report static and active resource quantities in one stable schema."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    active = getattr(module, "active_parameters_per_token", trainable)
    parameter_bytes = total * dtype_bytes
    optimizer_bytes = trainable * dtype_bytes * optimizer_state_multiplier
    route_flops = tokens * max(memory_slots, 0) * (2 * max(sequence_length, 1))
    sparse_flops = tokens * max(top_k, 0) * max(int(active), 0)
    kv_cache_bytes = tokens * max(sequence_length, 1) * dtype_bytes * 2
    memory_bank = sum(parameter.numel() for name, parameter in module.named_parameters() if any(token in name.lower() for token in ("key", "value", "expert", "basis", "token"))) * dtype_bytes
    return {
        "active_parameter_count": float(active),
        "total_parameter_count": float(total),
        "total_parameters": float(total),
        "trainable_parameters": float(trainable),
        "active_parameters_per_token": float(active),
        "parameter_bytes": float(parameter_bytes),
        "optimizer_state_bytes": float(optimizer_bytes),
        "KV_cache_bytes": float(kv_cache_bytes),
        "memory_bank_bytes": float(memory_bank),
        "estimated_static_bytes": float(parameter_bytes + optimizer_bytes),
        "estimated_router_flops": float(route_flops),
        "estimated_active_expert_flops": float(sparse_flops),
        "tokens": float(tokens),
        "memory_slots": float(memory_slots),
        "top_k": float(top_k),
    }


def finite_metrics(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (float, int)) and not torch.isfinite(torch.tensor(float(value))):
            return False
    return True


@torch.no_grad()
def measure_forward(
    module: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    repeats: int = 3,
    warmup: int = 1,
    device: str = "auto",
) -> dict[str, float | str]:
    """Measure a small forward pass with the same schema on CPU or CUDA."""
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    target = torch.device(resolved)
    module = module.to(target).eval()
    query = torch.randn(batch_size, sequence_length, d_model, device=target)
    for _ in range(warmup):
        module(query)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        module(query)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        durations.append((time.perf_counter() - start) * 1000.0)
    median_ms = float(torch.tensor(durations).median())
    peak = float(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0.0
    tokens = batch_size * sequence_length
    return {
        "device": str(target),
        "forward_median_ms": median_ms,
        "forward_min_ms": min(durations),
        "measured_forward_ms": median_ms,
        "throughput_tokens_per_sec": tokens / max(median_ms / 1000.0, 1e-12),
        "tokens_or_samples_per_second": tokens / max(median_ms / 1000.0, 1e-12),
        "peak_vram_bytes": peak,
        "peak_vram": peak,
    }


def measure_forward_backward(
    module: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    repeats: int = 3,
    device: str = "auto",
) -> dict[str, float | str]:
    """Measure forward/backward latency and peak memory for systems reports."""
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    target = torch.device(resolved)
    module = module.to(target).train()
    durations_forward: list[float] = []
    durations_backward: list[float] = []
    for _ in range(max(repeats, 1)):
        query = torch.randn(batch_size, sequence_length, d_model, device=target, requires_grad=True)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        start = time.perf_counter()
        output = module(query)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        durations_forward.append((time.perf_counter() - start) * 1000.0)
        module.zero_grad(set_to_none=True)
        start = time.perf_counter()
        output.square().mean().backward()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        durations_backward.append((time.perf_counter() - start) * 1000.0)
    forward_ms = float(torch.tensor(durations_forward).median())
    backward_ms = float(torch.tensor(durations_backward).median())
    peak = float(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0.0
    tokens = batch_size * sequence_length
    return {
        "device": str(target),
        "measured_forward_ms": forward_ms,
        "measured_backward_ms": backward_ms,
        "tokens_or_samples_per_second": tokens / max((forward_ms + backward_ms) / 1000.0, 1e-12),
        "peak_vram": peak,
    }
