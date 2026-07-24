from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BoundedDyck2Dataset(Dataset[dict[str, torch.Tensor]]):
    """Balanced two-bracket strings with a bounded maximum nesting depth."""

    def __init__(
        self,
        size: int = 1000,
        max_depth: int = 8,
        min_depth: int = 1,
        seed: int = 0,
        sequence_length: int | None = None,
    ) -> None:
        if size < 1 or not 1 <= min_depth <= max_depth:
            raise ValueError("Dyck size and depth bounds are invalid.")
        self.size = size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.seed = seed
        self.bos, self.open_a, self.close_a, self.open_b, self.close_b = range(5)
        self.vocab_size = 5
        self.sequence_length = sequence_length or (2 * max_depth)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        target_depth = int(rng.integers(self.min_depth, self.max_depth + 1))
        stack: list[int] = []
        tokens: list[int] = []
        while stack or len(tokens) < 2 * target_depth:
            can_open = len(stack) < target_depth and len(tokens) < 2 * target_depth - 1
            can_close = bool(stack)
            if can_open and (not can_close or rng.random() < 0.55):
                bracket = int(rng.integers(0, 2))
                stack.append(bracket)
                tokens.append(self.open_a if bracket == 0 else self.open_b)
            elif can_close:
                bracket = stack.pop()
                tokens.append(self.close_a if bracket == 0 else self.close_b)
        if len(tokens) < self.sequence_length:
            tokens.extend([self.close_a] * (self.sequence_length - len(tokens)))
        tokens = tokens[: self.sequence_length]
        full = np.asarray([self.bos] + tokens, dtype=np.int64)
        inputs = full[:-1]
        targets = full[1:]
        return {
            "inputs": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "loss_mask": torch.ones_like(torch.from_numpy(targets), dtype=torch.float32),
            "metadata": torch.full_like(torch.from_numpy(inputs), target_depth),
        }
