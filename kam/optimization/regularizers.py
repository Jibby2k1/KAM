from __future__ import annotations

import torch
from torch import Tensor


def coverage_loss(data: Tensor, supports: Tensor, metric: Tensor | None = None) -> Tensor:
    distances = torch.cdist(data.float(), supports.float()).square()
    if metric is not None:
        distances = distances * metric.diagonal().mean().clamp_min(1e-8)
    return distances.min(dim=-1).values.mean()


def repulsion_loss(supports: Tensor, rho: float = 1.0) -> Tensor:
    distances = torch.cdist(supports.float(), supports.float()).square()
    mask = ~torch.eye(supports.shape[0], device=supports.device, dtype=torch.bool)
    return torch.exp(-distances / max(rho**2, 1e-8))[mask].mean() if mask.any() else supports.sum() * 0


def load_balance_loss(assignments: Tensor, num_supports: int) -> Tensor:
    counts = torch.bincount(assignments.reshape(-1), minlength=num_supports).float()
    probabilities = counts / counts.sum().clamp_min(1e-12)
    return (probabilities - 1 / num_supports).square().mean()


def drift_loss(new_output: Tensor, old_output: Tensor) -> Tensor:
    if new_output.shape != old_output.shape:
        raise ValueError("drift outputs must have matching shapes")
    return (new_output - old_output).square().mean()


def metric_conditioning_loss(metric: Tensor, min_singular: float = 0.1, max_singular: float = 10.0) -> Tensor:
    singular = torch.linalg.svdvals(metric)
    return torch.relu(min_singular - singular).square().mean() + torch.relu(singular - max_singular).square().mean()


def weighted_regularization(terms: dict[str, Tensor], weights: dict[str, float] | None = None) -> Tensor:
    weights = weights or {}
    result = next(iter(terms.values())).sum() * 0
    for name, term in terms.items():
        result = result + float(weights.get(name, 0.0)) * term
    return result


__all__ = ["coverage_loss", "drift_loss", "load_balance_loss", "metric_conditioning_loss", "repulsion_loss", "weighted_regularization"]
