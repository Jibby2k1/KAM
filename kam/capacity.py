from __future__ import annotations

from typing import Any

from torch import nn


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def parameter_match_error(left: nn.Module, right: nn.Module) -> float:
    left_count = parameter_count(left)
    right_count = parameter_count(right)
    return abs(left_count - right_count) / max(left_count, right_count, 1)


def approximate_flops(spec: dict[str, Any], sequence_length: int, *, batch_size: int = 1) -> int:
    """Conservative forward FLOP estimate for comparison tables."""
    width = int(spec["d_model"])
    heads = int(spec["num_heads"])
    layers = int(spec.get("num_layers", 1))
    supports = int(spec.get("num_supports", 0))
    window = spec.get("context_window") or sequence_length
    context_pairs = sequence_length * min(sequence_length, int(window))
    memory_pairs = sequence_length * supports if spec.get("memory_score") else 0
    projection = 12 * sequence_length * width * width
    attention = 2 * (context_pairs + memory_pairs) * width
    return int(batch_size * layers * (projection + attention))
