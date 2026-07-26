from __future__ import annotations

import torch
from torch import Tensor


def relative_drift(new: Tensor, old: Tensor, eps: float = 1e-8) -> float:
    if new.shape != old.shape:
        raise ValueError("drift tensors must have matching shapes")
    return float((new - old).norm() / (old.norm() + eps))


def feature_drift(new_features: Tensor, old_features: Tensor, eps: float = 1e-8) -> float:
    return relative_drift(new_features, old_features, eps)


def function_drift(new_output: Tensor, old_output: Tensor, eps: float = 1e-8) -> float:
    return relative_drift(new_output, old_output, eps)


def support_drift(new_keys: Tensor, old_keys: Tensor, eps: float = 1e-8) -> dict[str, float]:
    if new_keys.ndim != 2 or old_keys.shape != new_keys.shape:
        raise ValueError("support banks must have shape [supports, features]")
    displacement = (new_keys - old_keys).norm(dim=-1)
    return {
        "mean_support_drift": float(displacement.mean()),
        "max_support_drift": float(displacement.max()),
        "relative_support_drift": relative_drift(new_keys, old_keys, eps),
    }


__all__ = ["feature_drift", "function_drift", "relative_drift", "support_drift"]
