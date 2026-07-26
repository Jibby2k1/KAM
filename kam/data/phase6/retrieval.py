from __future__ import annotations

import torch
from torch import Tensor


def variable_copy(batch: int = 32, payload_length: int = 8, vocab_size: int = 32, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    payload = torch.randint(vocab_size, (batch, payload_length), generator=generator)
    delimiter = torch.full((batch, 1), vocab_size - 1, dtype=torch.long)
    inputs = torch.cat((payload, delimiter, torch.zeros_like(payload)), dim=1)
    targets = torch.cat((torch.full_like(payload, -100), torch.full_like(delimiter, -100), payload), dim=1)
    return inputs, targets


def mqar(batch: int = 32, pairs: int = 4, sequence_length: int = 64, vocab_size: int = 64, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(vocab_size, (batch, sequence_length), generator=generator)
    targets = torch.full_like(inputs, -100)
    for pair in range(pairs):
        key = torch.randint(vocab_size, (batch,), generator=generator)
        value = torch.randint(vocab_size, (batch,), generator=generator)
        position = min(sequence_length - 1, 2 * pair + 1)
        query_position = sequence_length - pairs + pair
        inputs[:, position] = key
        inputs[:, min(position + 1, sequence_length - 1)] = value
        inputs[:, query_position] = key
        targets[:, query_position] = value
    return inputs, targets


def associative_recall(batch: int = 32, items: int = 8, distractors: int = 16, vocab_size: int = 64, seed: int = 0) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randint(vocab_size, (batch, items), generator=generator)
    values = torch.randint(vocab_size, (batch, items), generator=generator)
    noise = torch.randint(vocab_size, (batch, distractors), generator=generator)
    query_index = torch.randint(items, (batch,), generator=generator)
    query = keys.gather(1, query_index.unsqueeze(-1))
    target = values.gather(1, query_index.unsqueeze(-1))
    return torch.cat((keys, values, noise, query), dim=1), target.squeeze(-1)


def bounded_dyck(batch: int = 32, depth: int = 4, vocab_size: int = 4, seed: int = 0) -> tuple[Tensor, Tensor]:
    del vocab_size, seed
    sequence = torch.tensor(([1] * depth) + ([2] * depth), dtype=torch.long).repeat(batch, 1)
    return sequence, torch.roll(sequence, shifts=-1, dims=1)


__all__ = ["associative_recall", "bounded_dyck", "mqar", "variable_copy"]
