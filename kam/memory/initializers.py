from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def _validate(data: Tensor, num_supports: int, seed: int) -> tuple[Tensor, int]:
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must have shape [samples, features]")
    if num_supports <= 0:
        raise ValueError("num_supports must be positive")
    generator = torch.Generator(device=data.device).manual_seed(seed)
    return data, min(num_supports, data.shape[0]) if num_supports <= data.shape[0] else num_supports


def fixed_random(data: Tensor, num_supports: int, seed: int = 0) -> Tensor:
    data, _ = _validate(data, num_supports, seed)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    keys = torch.randn(num_supports, data.shape[-1], generator=generator, device=data.device, dtype=data.dtype)
    return torch.nn.functional.normalize(keys, dim=-1)


def fixed_data_sample(data: Tensor, num_supports: int, seed: int = 0) -> Tensor:
    data, _ = _validate(data, num_supports, seed)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    indices = torch.randperm(data.shape[0], generator=generator, device=data.device)[:num_supports]
    if indices.numel() < num_supports:
        extra = torch.randint(data.shape[0], (num_supports - indices.numel(),), generator=generator, device=data.device)
        indices = torch.cat((indices, extra))
    return data[indices].clone()


def kmeans(data: Tensor, num_supports: int, seed: int = 0, iterations: int = 12) -> Tensor:
    data, _ = _validate(data, num_supports, seed)
    centers = fixed_data_sample(data, num_supports, seed)
    for _ in range(iterations):
        distances = torch.cdist(data.float(), centers.float())
        assignments = distances.argmin(-1)
        updated = centers.clone()
        for index in range(num_supports):
            selected = data[assignments == index]
            if selected.numel():
                updated[index] = selected.mean(0)
        if torch.allclose(updated, centers, atol=1e-6, rtol=1e-5):
            break
        centers = updated
    return centers


def farthest_point(data: Tensor, num_supports: int, seed: int = 0) -> Tensor:
    data, _ = _validate(data, num_supports, seed)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    first = int(torch.randint(data.shape[0], (), generator=generator, device=data.device))
    selected = [first]
    distances = torch.cdist(data[[first]].float(), data.float()).squeeze(0)
    for _ in range(1, num_supports):
        index = int(distances.argmax())
        selected.append(index)
        distances = torch.minimum(distances, torch.cdist(data[[index]].float(), data.float()).squeeze(0))
    return data[selected].clone()


def initialize_keys(data: Tensor, num_supports: int, mode: str, seed: int = 0) -> Tensor:
    modes: dict[str, object] = {
        "fixed_random": fixed_random,
        "fixed_data_sample": fixed_data_sample,
        "fixed_kmeans": kmeans,
        "fixed_farthest_point": farthest_point,
    }
    try:
        function = modes[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported key initializer: {mode}") from exc
    return function(data, num_supports, seed=seed)  # type: ignore[operator]


__all__ = ["farthest_point", "fixed_data_sample", "fixed_random", "initialize_keys", "kmeans"]
