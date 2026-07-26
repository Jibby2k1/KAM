from __future__ import annotations

import torch
from torch import Tensor

from .ridge import RidgeResult, streaming_rls


def recursive_least_squares(features: Tensor, targets: Tensor, regularization: float = 1e-2) -> RidgeResult:
    return streaming_rls(features, targets, regularization=regularization)


class RLSReadout:
    """Predict-then-update scalar/vector RLS readout for online streams."""

    def __init__(self, feature_dim: int, output_dim: int = 1, regularization: float = 1.0) -> None:
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.beta = torch.zeros(feature_dim, output_dim)
        self.precision = torch.eye(feature_dim) / max(regularization, 1e-12)

    def predict(self, features: Tensor) -> Tensor:
        beta = self.beta.to(device=features.device, dtype=features.dtype)
        return features @ beta

    @torch.no_grad()
    def update(self, features: Tensor, targets: Tensor) -> Tensor:
        x = features.reshape(-1)
        y = targets.reshape(-1)
        precision = self.precision.to(device=x.device, dtype=x.dtype)
        beta = self.beta.to(device=x.device, dtype=x.dtype)
        px = precision @ x
        gain = px / (1.0 + x @ px)
        error = y - x @ beta[:, 0]
        beta[:, 0] += gain * error
        self.precision = precision - torch.outer(gain, x @ precision)
        self.beta = beta
        return error


__all__ = ["RLSReadout", "recursive_least_squares"]
