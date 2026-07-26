from __future__ import annotations

import torch
from torch import Tensor


class NLMSReadout:
    """Normalized-LMS predict-then-update readout."""

    def __init__(self, feature_dim: int, output_dim: int = 1, step_size: float = 0.5, eps: float = 1e-8) -> None:
        self.weights = torch.zeros(feature_dim, output_dim)
        self.step_size = step_size
        self.eps = eps

    def predict(self, features: Tensor) -> Tensor:
        return features @ self.weights.to(device=features.device, dtype=features.dtype)

    @torch.no_grad()
    def update(self, features: Tensor, targets: Tensor) -> Tensor:
        x = features.reshape(-1)
        target = targets.reshape(-1)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        error = target - x @ weights[:, 0]
        weights[:, 0] += self.step_size * error * x / (x.square().sum() + self.eps)
        self.weights = weights
        return error


__all__ = ["NLMSReadout"]
