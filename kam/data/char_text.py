from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Character data should contain at least 100 characters.")
    return text


class CharacterDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic character-level next-token blocks from a local text file."""

    def __init__(
        self,
        text: str,
        sequence_length: int = 128,
        stride: int | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.characters = list(vocabulary) if vocabulary is not None else sorted(set(text))
        self.stoi = {character: index for index, character in enumerate(self.characters)}
        self.itos = {index: character for character, index in self.stoi.items()}
        unknown = sorted(set(text) - set(self.characters))
        if unknown:
            raise ValueError(f"Text contains characters absent from the supplied vocabulary: {unknown}")
        self.encoded = np.asarray([self.stoi[character] for character in text], dtype=np.int64)
        if len(self.encoded) <= sequence_length + 1:
            raise ValueError("Text is too short for the requested sequence length.")
        self.starts = list(range(0, len(self.encoded) - sequence_length - 1, self.stride))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.starts[index]
        block = self.encoded[start : start + self.sequence_length + 1]
        inputs = torch.from_numpy(block[:-1])
        targets = torch.from_numpy(block[1:])
        return {
            "inputs": inputs,
            "targets": targets,
            "loss_mask": torch.ones_like(targets, dtype=torch.float32),
            "metadata": torch.full_like(targets, -1),
        }
