from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RidgeResult:
    solution: Tensor
    condition_number: float
    residual_norm: float
    solver: str


def _as_2d_targets(targets: Tensor) -> tuple[Tensor, bool]:
    if targets.ndim == 1:
        return targets.unsqueeze(-1), True
    if targets.ndim != 2:
        raise ValueError("targets must have shape [samples] or [samples, outputs]")
    return targets, False


def ridge_solve(features: Tensor, targets: Tensor, regularization: float = 1e-4, solver: str = "cholesky") -> RidgeResult:
    """Solve a small ridge problem with a transparent conditioning report."""
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must share a sample dimension")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    y, squeeze = _as_2d_targets(targets)
    x = features
    gram = x.T @ x + float(regularization) * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    rhs = x.T @ y
    if solver == "cholesky":
        factor = torch.linalg.cholesky(gram)
        solution = torch.cholesky_solve(rhs, factor)
    elif solver in {"solve", "cg", "blockwise"}:
        solution = torch.linalg.solve(gram, rhs)
    else:
        raise ValueError(f"unknown ridge solver: {solver}")
    predicted = x @ solution
    condition = float(torch.linalg.cond(gram).detach())
    residual = float((predicted - y).square().sum().sqrt().detach())
    if squeeze:
        solution = solution[:, 0]
    return RidgeResult(solution, condition, residual, solver)


def ridge_objective(features: Tensor, targets: Tensor, solution: Tensor, regularization: float = 1e-4) -> Tensor:
    y, _ = _as_2d_targets(targets)
    beta = solution.unsqueeze(-1) if solution.ndim == 1 else solution
    return (features @ beta - y).square().mean() + regularization * beta.square().mean()


def streaming_rls(features: Tensor, targets: Tensor, regularization: float = 1e-2) -> RidgeResult:
    """Reference recursive least-squares solve for tiny online fixtures."""
    if features.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("features and targets must share a sample dimension")
    y, squeeze = _as_2d_targets(targets)
    d = features.shape[1]
    precision = torch.eye(d, device=features.device, dtype=features.dtype) / max(regularization, 1e-12)
    beta = torch.zeros(d, y.shape[1], device=features.device, dtype=features.dtype)
    for x_row, y_row in zip(features, y):
        px = precision @ x_row
        gain = px / (1.0 + x_row @ px)
        error = y_row - x_row @ beta
        beta = beta + gain.unsqueeze(-1) * error.unsqueeze(0)
        precision = precision - torch.outer(gain, x_row @ precision)
    condition = float(torch.linalg.cond(precision).detach())
    residual = float((features @ beta - y).square().sum().sqrt().detach())
    if squeeze:
        beta = beta[:, 0]
    return RidgeResult(beta, condition, residual, "streaming_rls")
