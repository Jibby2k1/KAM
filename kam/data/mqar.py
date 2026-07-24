from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MQARDataset(Dataset[dict[str, torch.Tensor]]):
    """Multi-query associative recall with deterministic distractors.

    Each example contains key/value bindings followed by queries. The loss is
    masked to the value token immediately following every query key.
    """

    def __init__(
        self,
        size: int = 1000,
        sequence_length: int = 128,
        num_bindings: int = 8,
        num_queries: int = 4,
        vocab_size: int = 128,
        seed: int = 0,
    ) -> None:
        if min(size, sequence_length, num_bindings, num_queries) < 1:
            raise ValueError("MQAR dimensions must be positive.")
        if vocab_size < 2 * num_bindings + 4:
            raise ValueError("vocab_size is too small for unique keys and values.")
        minimum = 1 + 2 * num_bindings + 2 * num_queries
        if sequence_length < minimum:
            raise ValueError(f"sequence_length must be at least {minimum}.")
        self.size = size
        self.sequence_length = sequence_length
        self.num_bindings = num_bindings
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.seed = seed
        self.bos = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        keys = np.arange(1, self.num_bindings + 1, dtype=np.int64)
        values = rng.choice(
            np.arange(self.num_bindings + 1, self.vocab_size, dtype=np.int64),
            size=self.num_bindings,
            replace=False,
        )
        order = rng.permutation(self.num_bindings)
        body: list[int] = []
        for item in order:
            body.extend([int(keys[item]), int(values[item])])

        remaining = self.sequence_length - len(body) - 2 * self.num_queries
        if remaining:
            distractors = rng.integers(1, self.vocab_size, size=remaining)
            body.extend(int(token) for token in distractors)
        queries = rng.integers(0, self.num_bindings, size=self.num_queries)
        target_positions: list[int] = []
        for query in queries:
            body.append(int(keys[query]))
            target_positions.append(len(body))
            body.append(int(values[query]))

        full = np.asarray([self.bos] + body, dtype=np.int64)
        if len(full) != self.sequence_length + 1:
            raise RuntimeError("MQAR example has an unexpected length.")
        inputs = full[:-1]
        targets = full[1:]
        mask = np.zeros_like(targets, dtype=np.float32)
        for full_index in target_positions:
            mask[full_index - 1] = 1.0
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "loss_mask": torch.from_numpy(mask),
            "metadata": torch.full_like(torch.from_numpy(inputs), -1),
        }
