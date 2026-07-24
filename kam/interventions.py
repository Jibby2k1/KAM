from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn

from .diagnostics import support_utilization


def _zero_update(output: tuple[Tensor, Tensor | None], *, zero_weights: bool) -> tuple[Tensor, Tensor | None]:
    update, weights = output
    if zero_weights and weights is not None:
        weights = torch.zeros_like(weights)
    return torch.zeros_like(update), weights


@contextmanager
def ablate_branches(
    model: nn.Module,
    *,
    context: bool = False,
    memory: bool = False,
) -> Iterator[nn.Module]:
    """Ablate context and/or memory updates while preserving output shapes.

    Memory weights are zeroed during memory ablation so route-based readouts do
    not receive an unablated side channel.
    """
    handles = []
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        if context and getattr(block, "context", None) is not None:
            handles.append(block.context.register_forward_hook(lambda _m, _i, out: _zero_update(out, zero_weights=False)))
        if memory and getattr(block, "memory", None) is not None:
            handles.append(block.memory.register_forward_hook(lambda _m, _i, out: _zero_update(out, zero_weights=True)))
    try:
        yield model
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def perturb_memory(
    model: nn.Module,
    *,
    key_noise: float = 0.0,
    value_noise: float = 0.0,
    zero_keys: bool = False,
    zero_values: bool = False,
    seed: int = 0,
) -> Iterator[nn.Module]:
    """Perturb learned memory keys/values and restore them on exit."""
    generators: list[tuple[nn.Parameter, Tensor]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for block in getattr(model, "blocks", []):
        memory = getattr(block, "memory", None)
        if memory is None:
            continue
        for name, scale, zero in (
            ("memory_keys", key_noise, zero_keys),
            ("memory_values", value_noise, zero_values),
        ):
            parameter = getattr(memory, name)
            original = parameter.detach().clone()
            generators.append((parameter, original))
            with torch.no_grad():
                if zero:
                    parameter.zero_()
                elif scale:
                    noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                    parameter.add_(noise.to(parameter.device) * float(scale))
    try:
        yield model
    finally:
        with torch.no_grad():
            for parameter, original in generators:
                parameter.copy_(original)


@contextmanager
def support_mask(model: nn.Module, mask: Tensor) -> Iterator[nn.Module]:
    """Apply one support deletion mask to every memory block."""
    old_masks: list[tuple[nn.Module, Tensor | None]] = []
    for block in getattr(model, "blocks", []):
        memory = getattr(block, "memory", None)
        if memory is None or not hasattr(memory, "set_support_mask"):
            continue
        old_masks.append((memory, getattr(memory, "support_mask", None)))
        memory.set_support_mask(mask)
    try:
        yield model
    finally:
        for memory, old_mask in old_masks:
            memory.set_support_mask(old_mask)


def support_rankings(weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return descending and ascending support indices from mean routing mass."""
    if weights.ndim != 4:
        raise ValueError("weights must have shape [B, H, T, M].")
    mass = weights.detach().float().mean(dim=(0, 1, 2))
    return torch.argsort(mass, descending=True), torch.argsort(mass, descending=False)


def frozen_ridge_probe(features: Tensor, targets: Tensor, *, ridge: float = 1e-3) -> dict[str, float]:
    """Fit/evaluate a frozen linear probe with a deterministic half split."""
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features must be [N,F] and targets must be [N,O].")
    split = max(1, features.shape[0] // 2)
    train_x, test_x = features[:split].float(), features[split:].float()
    train_y, test_y = targets[:split].float(), targets[split:].float()
    if test_x.shape[0] == 0:
        test_x, test_y = train_x, train_y
    ones = torch.ones(train_x.shape[0], 1, dtype=train_x.dtype, device=train_x.device)
    train_aug = torch.cat([train_x, ones], dim=-1)
    gram = train_aug.T @ train_aug + ridge * torch.eye(train_aug.shape[1], device=train_aug.device)
    weights = torch.linalg.solve(gram, train_aug.T @ train_y)
    test_aug = torch.cat([test_x, torch.ones(test_x.shape[0], 1, device=test_x.device)], dim=-1)
    prediction = test_aug @ weights
    error = prediction - test_y
    return {
        "probe_mse": float(error.square().mean()),
        "probe_mae": float(error.abs().mean()),
        "probe_train_samples": float(train_x.shape[0]),
        "probe_test_samples": float(test_x.shape[0]),
    }


def summarize_support(weights: Tensor) -> dict[str, float]:
    return support_utilization(weights)
