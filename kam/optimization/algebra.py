from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .ridge import RidgeResult, ridge_solve


@dataclass
class ParameterPartition:
    geometry: list[str]
    algebra: list[str]
    backbone: list[str]


def partition_parameters(module: nn.Module) -> ParameterPartition:
    """Expose geometry/algebra/backbone groups without relying on optimizer order."""
    geometry: list[str] = []
    algebra: list[str] = []
    backbone: list[str] = []
    for name, parameter in module.named_parameters():
        if any(token in name.lower() for token in ("key", "geometry", "codebook", "metric")):
            geometry.append(name)
        elif any(token in name.lower() for token in ("expert", "value", "bias", "coefficient", "gate")):
            algebra.append(name)
        else:
            backbone.append(name)
    return ParameterPartition(geometry=geometry, algebra=algebra, backbone=backbone)


def solve_algebra(features: Tensor, targets: Tensor, *, solver: str = "cholesky", regularization: float = 1e-4) -> RidgeResult:
    """Shared entry point for Cholesky/CG/blockwise/RLS algebra solves."""
    if solver == "rls":
        from .ridge import streaming_rls

        return streaming_rls(features, targets, regularization)
    return ridge_solve(features, targets, regularization=regularization, solver=solver)


def algebra_transport(
    old_features: Tensor,
    new_features: Tensor,
    old_solution: Tensor,
    *,
    regularization: float = 1e-4,
) -> dict[str, Tensor | float]:
    """Transport algebra by matching old anchor predictions after geometry motion."""
    target = old_features @ (old_solution.unsqueeze(-1) if old_solution.ndim == 1 else old_solution)
    solved = ridge_solve(new_features, target, regularization=regularization)
    new_solution = solved.solution
    new_prediction = new_features @ (new_solution.unsqueeze(-1) if new_solution.ndim == 1 else new_solution)
    error = float((new_prediction - target).square().mean().sqrt().detach())
    return {"solution": new_solution, "transport_error": error, "condition_number": solved.condition_number}


__all__ = ["ParameterPartition", "algebra_transport", "partition_parameters", "solve_algebra"]
