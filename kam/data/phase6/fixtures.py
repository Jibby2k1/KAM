from __future__ import annotations

import torch
from torch import Tensor


def make_dynamics_batch(batch: int = 8, length: int = 32, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, generator=generator)
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0]
    for step in range(1, length):
        y[:, step] = 0.85 * y[:, step - 1] + 0.1 * x[:, step]
    return x.unsqueeze(-1), y.unsqueeze(-1)


def make_retrieval_batch(batch: int = 8, length: int = 16, vocab_size: int = 16, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(vocab_size, (batch, length), generator=generator)
    targets = torch.roll(tokens, shifts=-1, dims=1)
    return tokens, targets


def make_symbolic_batch(batch: int = 8, length: int = 32, modulus: int = 7, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    regimes = torch.randint(2, (batch,), generator=generator)
    tokens = torch.zeros(batch, length, dtype=torch.long)
    tokens[:, 0] = torch.randint(modulus, (batch,), generator=generator)
    for step in range(1, length):
        increment = torch.where(regimes == 0, 1, 2)
        tokens[:, step] = (tokens[:, step - 1] + increment) % modulus
    return tokens, torch.roll(tokens, shifts=-1, dims=1), regimes
