from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


DEFAULT_TEXT = "the small memory model reads a stream and predicts the next token . "
DEFAULT_CORPUS_PATHS = (Path("data/tinyshakespeare.txt"), Path("data/sample_text.txt"))


def load_text_tokens(path: str | Path | None = None) -> tuple[Tensor, dict[str, int]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        corpus_path = next((candidate for candidate in DEFAULT_CORPUS_PATHS if candidate.exists() and candidate.stat().st_size > 0), None)
        text = corpus_path.read_text(encoding="utf-8") if corpus_path else DEFAULT_TEXT
    vocabulary = {character: index for index, character in enumerate(sorted(set(text)))}
    tokens = torch.tensor([vocabulary[character] for character in text], dtype=torch.long)
    return tokens, vocabulary


def language_batches(tokens: Tensor, *, batch_size: int = 8, sequence_length: int = 32, seed: int = 0):
    if tokens.numel() <= sequence_length + 1:
        raise ValueError("language corpus must exceed sequence length")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(tokens.numel() - sequence_length - 1, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + sequence_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts])
    return inputs, targets


__all__ = ["DEFAULT_TEXT", "language_batches", "load_text_tokens"]
