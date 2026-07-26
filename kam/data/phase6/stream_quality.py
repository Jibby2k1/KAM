from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from kam.data.stream_quality import assess_stream_quality, require_stable_stream, stream_quality_checks


def scheduled_stream(streams: dict[str, tuple[Tensor, Tensor]], order: tuple[str, ...] = ("A", "B", "A", "C", "A")) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.cat([streams[name][0] for name in order], dim=0)
    targets = torch.cat([streams[name][1] for name in order], dim=0)
    labels = torch.cat([torch.full((streams[name][0].shape[0],), index, dtype=torch.long) for index, name in enumerate(order)])
    return features, targets, labels


def prequential_evaluate(
    features: Tensor,
    targets: Tensor,
    *,
    predict: Callable[[Tensor], Tensor],
    update: Callable[[Tensor, Tensor], None] | None = None,
) -> dict[str, float | list[float]]:
    """Evaluate with predict → score → reveal → update ordering."""
    errors: list[float] = []
    for feature, target in zip(features, targets):
        prediction = predict(feature)
        errors.append(float((prediction.detach() - target).square().mean()))
        if update is not None:
            update(feature, target)
    if not errors:
        raise ValueError("prequential stream cannot be empty")
    cut = max(1, len(errors) // 3)
    return {
        "global_nmse": float(np.mean(errors) / (np.var(targets.detach().cpu().numpy()) + 1e-12)),
        "early_nmse": float(np.mean(errors[:cut]) / (np.var(targets.detach().cpu().numpy()) + 1e-12)),
        "late_nmse": float(np.mean(errors[-cut:]) / (np.var(targets.detach().cpu().numpy()) + 1e-12)),
        "errors": errors,
    }


__all__ = ["assess_stream_quality", "prequential_evaluate", "require_stable_stream", "scheduled_stream", "stream_quality_checks"]
