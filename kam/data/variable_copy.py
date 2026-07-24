from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VariableCopyLanguageDataset(Dataset[dict[str, torch.Tensor]]):
    """Copy task with per-example payload lengths and no absolute-length shortcut."""

    def __init__(
        self,
        size: int = 10000,
        min_payload_length: int = 4,
        max_payload_length: int = 32,
        alphabet_size: int = 8,
        seed: int = 0,
    ) -> None:
        if size < 1 or min_payload_length < 1 or max_payload_length < min_payload_length or alphabet_size < 2:
            raise ValueError("invalid variable-copy dimensions")
        self.size = size
        self.min_payload_length = min_payload_length
        self.max_payload_length = max_payload_length
        self.alphabet_size = alphabet_size
        self.seed = seed
        self.bos = alphabet_size
        self.separator = alphabet_size + 1
        self.eos = alphabet_size + 2
        self.pad = alphabet_size + 3
        self.vocab_size = alphabet_size + 4
        self.max_sequence_length = 2 * max_payload_length + 2

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        length = int(rng.integers(self.min_payload_length, self.max_payload_length + 1))
        payload = rng.integers(0, self.alphabet_size, size=length, dtype=np.int64)
        full = np.concatenate([
            np.asarray([self.bos], dtype=np.int64),
            payload,
            np.asarray([self.separator], dtype=np.int64),
            payload,
            np.asarray([self.eos], dtype=np.int64),
        ])
        targets = full[1:]
        mask = np.zeros_like(targets, dtype=np.float32)
        start = length + 1
        mask[start : start + length] = 1.0
        return {
            "inputs": torch.from_numpy(full[:-1]),
            "targets": torch.from_numpy(targets),
            "loss_mask": torch.from_numpy(mask),
            "metadata": torch.tensor(length, dtype=torch.long),
        }


def variable_copy_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(item["inputs"].shape[0] for item in batch)
    pad_token = max(max(int(item["inputs"].max()), int(item["targets"].max())) for item in batch) + 1
    inputs = torch.full((len(batch), max_length), pad_token, dtype=torch.long)
    targets = torch.full((len(batch), max_length), pad_token, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_length), dtype=torch.float32)
    metadata = torch.zeros((len(batch),), dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["inputs"].shape[0]
        inputs[row, :length] = item["inputs"]
        targets[row, :length] = item["targets"]
        loss_mask[row, :length] = item["loss_mask"]
        metadata[row] = item["metadata"]
    return {"inputs": inputs, "targets": targets, "loss_mask": loss_mask, "metadata": metadata}
