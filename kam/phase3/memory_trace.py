"""Small, dependency-light diagnostics for learned memory support banks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def memory_bank_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return learned memory key/value parameters by their stable parameter names."""

    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("memory_keys") or name.endswith("memory_values")
    }


def set_memory_bank_trainable(model: nn.Module, trainable: bool) -> int:
    """Set key/value-bank gradients and return the number of affected parameters."""

    parameters = memory_bank_parameters(model)
    for parameter in parameters.values():
        parameter.requires_grad_(trainable)
    return sum(parameter.numel() for parameter in parameters.values())


def snapshot_memory_bank(model: nn.Module) -> dict[str, Tensor]:
    """Copy the memory bank to CPU so later drift is independent of autograd."""

    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in memory_bank_parameters(model).items()
    }


def _kind(name: str) -> str:
    return "key" if name.endswith("memory_keys") else "value"


def _safe_relative(numerator: Tensor, denominator: Tensor) -> float:
    return float(numerator.norm() / denominator.norm().clamp_min(1e-12))


def _support_drift(
    current: dict[str, Tensor],
    reference: dict[str, Tensor],
    kind: str,
) -> list[float]:
    current_parts: list[Tensor] = []
    reference_parts: list[Tensor] = []
    for name, value in current.items():
        if _kind(name) != kind or name not in reference:
            continue
        baseline = reference[name]
        if value.ndim < 2:
            continue
        # The support axis is the penultimate dimension for [heads, supports, width].
        current_parts.append(value.transpose(0, 1).reshape(value.shape[1], -1))
        reference_parts.append(baseline.transpose(0, 1).reshape(baseline.shape[1], -1))
    if not current_parts:
        return []
    deltas = torch.cat(
        [
            (current_value - reference_value).square().sum(dim=1, keepdim=True)
            for current_value, reference_value in zip(current_parts, reference_parts)
        ],
        dim=1,
    ).sum(dim=1).sqrt()
    norms = torch.cat(
        [reference_value.square().sum(dim=1, keepdim=True) for reference_value in reference_parts],
        dim=1,
    ).sum(dim=1).sqrt().clamp_min(1e-12)
    return [float(value) for value in (deltas / norms)]


def memory_bank_drift(
    model: nn.Module,
    initial: dict[str, Tensor],
    previous: dict[str, Tensor] | None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, Tensor]]:
    """Measure global and per-support movement from initialization and last trace."""

    current = snapshot_memory_bank(model)
    summary: dict[str, float] = {}
    support: dict[str, list[float]] = {}
    for kind in ("key", "value"):
        current_parts = [value for name, value in current.items() if _kind(name) == kind and name in initial]
        initial_parts = [initial[name] for name in current if _kind(name) == kind and name in initial]
        if not current_parts:
            continue
        current_flat = torch.cat([value.reshape(-1) for value in current_parts])
        initial_flat = torch.cat([value.reshape(-1) for value in initial_parts])
        summary[f"memory_{kind}_norm"] = float(current_flat.norm())
        summary[f"memory_{kind}_relative_drift"] = _safe_relative(current_flat - initial_flat, initial_flat)
        if previous is not None:
            previous_parts = [
                previous[name].reshape(-1)
                for name in current
                if _kind(name) == kind and name in previous
            ]
            previous_flat = torch.cat(previous_parts) if previous_parts else current_flat
            summary[f"memory_{kind}_step_delta"] = _safe_relative(current_flat - previous_flat, previous_flat)
        else:
            summary[f"memory_{kind}_step_delta"] = 0.0
        support[f"memory_{kind}_support_drift"] = _support_drift(current, initial, kind)
    return summary, support, current


@torch.no_grad()
def support_usage(model: nn.Module, probe_inputs: Tensor) -> dict[str, Any]:
    """Summarize the final block's attention over supports on a fixed probe batch."""

    try:
        _predictions, diagnostics = model(probe_inputs, return_weights=True)
    except (AttributeError, RuntimeError, TypeError):
        return {}
    if not diagnostics.memory_weights:
        return {}
    weights = diagnostics.memory_weights[-1].detach().float()
    mean_weights = weights.mean(dim=(0, 1, 2))
    row_entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "support_weights": [float(value) for value in mean_weights.cpu()],
        "support_entropy": float(row_entropy.cpu()),
        "effective_supports": float(torch.exp(row_entropy).cpu()),
        "top1_support_mass": float(mean_weights.max().cpu()),
        "support_count": int(mean_weights.numel()),
    }


def trace_row(
    model: nn.Module,
    initial: dict[str, Tensor],
    previous: dict[str, Tensor] | None,
    *,
    step: int,
    stage: str,
    memory_bank_trainable: bool,
    probe_inputs: Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create one compact summary row and optional support-level rows."""

    summary, support_drift, _current = memory_bank_drift(model, initial, previous)
    usage = support_usage(model, probe_inputs) if probe_inputs is not None else {}
    summary.update({key: value for key, value in usage.items() if key != "support_weights"})
    summary.update(
        {
            "step": int(step),
            "stage": stage,
            "memory_bank_trainable": bool(memory_bank_trainable),
        }
    )
    support_rows: list[dict[str, Any]] = []
    weights = usage.get("support_weights", [])
    count = max(len(weights), len(support_drift.get("memory_key_support_drift", [])), len(support_drift.get("memory_value_support_drift", [])))
    for support_index in range(count):
        support_rows.append(
            {
                "step": int(step),
                "stage": stage,
                "support": support_index,
                "mean_attention": float(weights[support_index]) if support_index < len(weights) else None,
                "key_relative_drift": (
                    float(support_drift["memory_key_support_drift"][support_index])
                    if support_index < len(support_drift.get("memory_key_support_drift", []))
                    else None
                ),
                "value_relative_drift": (
                    float(support_drift["memory_value_support_drift"][support_index])
                    if support_index < len(support_drift.get("memory_value_support_drift", []))
                    else None
                ),
            }
        )
    return summary, support_rows


def finite_trace_value(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
