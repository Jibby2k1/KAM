from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


def effective_support_count(weights: Tensor, dim: int = -1) -> Tensor:
    probabilities = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return 1.0 / probabilities.square().sum(dim=dim).clamp_min(1e-12)


def gini(values: Tensor) -> float:
    values = values.detach().float().flatten().clamp_min(0)
    if values.numel() == 0 or float(values.sum()) == 0.0:
        return 0.0
    sorted_values = values.sort().values
    index = torch.arange(1, sorted_values.numel() + 1, device=values.device, dtype=values.dtype)
    return float(((2 * index - sorted_values.numel() - 1) * sorted_values).sum() / (sorted_values.numel() * sorted_values.sum()))


def support_utilization(weights: Tensor, dead_threshold: float = 1e-4) -> dict[str, float]:
    """Return compact utilization diagnostics for [B,H,T,M] routing weights."""
    if weights.ndim != 4:
        raise ValueError("weights must have shape [B, H, T, M].")
    mean_mass = weights.mean(dim=(0, 1, 2))
    normalized = mean_mass / mean_mass.sum().clamp_min(1e-12)
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum()
    top1 = weights.mean(dim=(0, 1, 2)).argmax(dim=-1)
    top1_frequency = torch.bincount(top1[None], minlength=mean_mass.numel()).float().sum()
    return {
        "num_supports": float(mean_mass.numel()),
        "global_effective_supports": float(1.0 / normalized.square().sum().clamp_min(1e-12)),
        "mean_local_effective_supports": float(effective_support_count(weights).mean()),
        "routing_entropy": float(entropy),
        "dead_support_fraction": float((mean_mass <= dead_threshold).float().mean()),
        "usage_gini": gini(mean_mass),
        "top1_support": float(top1),
    }


def paired_deletion_curve(
    baseline_losses: Tensor,
    top_losses: dict[int, Tensor],
    random_losses: dict[int, Tensor],
    bottom_losses: dict[int, Tensor] | None = None,
) -> list[dict[str, float]]:
    """Normalize top/random/bottom deletion losses against the intact baseline."""
    base = float(baseline_losses.mean())
    rows: list[dict[str, float]] = []
    for count in sorted(top_losses):
        row = {
            "deletion_count": float(count),
            "baseline_loss": base,
            "top_loss": float(top_losses[count].mean()),
            "random_loss": float(random_losses[count].mean()),
        }
        row["top_delta"] = row["top_loss"] - base
        row["random_delta"] = row["random_loss"] - base
        if bottom_losses is not None and count in bottom_losses:
            row["bottom_loss"] = float(bottom_losses[count].mean())
            row["bottom_delta"] = row["bottom_loss"] - base
        rows.append(row)
    return rows


def support_regime_metrics(assignments: Tensor, regimes: Tensor) -> dict[str, float]:
    """Compute purity and entropy without assuming one support per regime."""
    assignments = assignments.detach().flatten().cpu().numpy()
    regimes = regimes.detach().flatten().cpu().numpy()
    valid = regimes >= 0
    assignments = assignments[valid]
    regimes = regimes[valid]
    if assignments.size == 0:
        return {"purity": float("nan"), "conditional_entropy": float("nan")}
    correct = 0
    entropy = 0.0
    for support in np.unique(assignments):
        labels = regimes[assignments == support]
        counts = np.bincount(labels)
        probabilities = counts[counts > 0] / len(labels)
        correct += int(counts.max())
        entropy += len(labels) * float(-(probabilities * np.log(probabilities)).sum())
    return {
        "purity": correct / len(assignments),
        "conditional_entropy": entropy / len(assignments),
    }
