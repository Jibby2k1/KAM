from __future__ import annotations

import torch
from torch import Tensor


def controlled_symbolic_regimes(batch: int = 32, length: int = 32, modulus: int = 7, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    regime = torch.randint(3, (batch,), generator=generator)
    tokens = torch.zeros(batch, length, dtype=torch.long)
    tokens[:, 0] = torch.randint(modulus, (batch,), generator=generator)
    increments = torch.tensor([1, 2, 3])
    for step in range(1, length):
        tokens[:, step] = (tokens[:, step - 1] + increments[regime]) % modulus
    return tokens, torch.roll(tokens, shifts=-1, dims=1), regime


def regime_purity(assignments: Tensor, regimes: Tensor) -> float:
    assignments = assignments.reshape(-1)
    regimes = regimes.reshape(-1)
    correct = 0
    for support in assignments.unique():
        selected = regimes[assignments == support]
        if selected.numel():
            correct += int(torch.bincount(selected).max())
    return correct / max(regimes.numel(), 1)


__all__ = ["controlled_symbolic_regimes", "regime_purity"]
