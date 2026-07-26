from __future__ import annotations

import torch
from torch import Tensor


def nearest_assignments(data: Tensor, supports: Tensor) -> Tensor:
    if data.ndim != 2 or supports.ndim != 2 or data.shape[-1] != supports.shape[-1]:
        raise ValueError("data and supports must have shape [samples, features]")
    return torch.cdist(data.float(), supports.float()).argmin(-1)


def dictionary_update(data: Tensor, supports: Tensor, *, replacement_threshold: float = 0.0) -> tuple[Tensor, dict[str, float]]:
    assignments = nearest_assignments(data, supports)
    updated = supports.clone()
    counts = torch.bincount(assignments, minlength=supports.shape[0])
    for index in range(supports.shape[0]):
        selected = data[assignments == index]
        if selected.numel():
            updated[index] = selected.mean(0)
        elif replacement_threshold > 0:
            distances = torch.cdist(data.float(), supports[index : index + 1].float()).squeeze(-1)
            if float(distances.min()) > replacement_threshold:
                updated[index] = data[distances.argmax()]
    return updated, {
        "coverage": float((counts > 0).float().mean()),
        "dead_support_fraction": float((counts == 0).float().mean()),
        "mean_assignment_distance": float(torch.cdist(data.float(), supports.float()).min(-1).values.mean()),
    }


__all__ = ["dictionary_update", "nearest_assignments"]
