from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .ridge import ridge_solve


def variable_projection_objective(
    features: Tensor,
    targets: Tensor,
    geometry: Tensor,
    regularization: float = 1e-4,
    mode: Literal["stopgrad", "implicit", "unrolled"] = "stopgrad",
) -> Tensor:
    """Small differentiable fixture for geometry-through-algebra tests.

    ``stopgrad`` matches the common envelope-theorem implementation.  The
    other labels currently share the exact solve but remain visible in the
    manifest so their gradient semantics can be extended independently.
    """
    if mode not in {"stopgrad", "implicit", "unrolled"}:
        raise ValueError("unknown variable-projection mode")
    transformed = features @ geometry
    solved = ridge_solve(transformed.detach() if mode == "stopgrad" else transformed, targets, regularization).solution
    beta = solved.detach() if mode == "stopgrad" else solved
    if targets.ndim == 1:
        prediction = transformed @ beta
    else:
        prediction = transformed @ beta
    return (prediction - targets).square().mean() + regularization * beta.square().mean()
