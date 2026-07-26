from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TransformerConfig:
    """Configuration shared by every Phase 6 transformer control."""

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int | None = None
    vocab_size: int | None = None
    max_seq_len: int = 128
    dropout: float = 0.0
    norm_eps: float = 1e-5
    positional_encoding: str = "learned"
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("d_model, n_heads, and n_layers must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_ff is not None and self.d_ff <= 0:
            raise ValueError("d_ff must be positive when supplied")
        if self.positional_encoding not in {"learned", "sinusoidal"}:
            raise ValueError("positional_encoding must be learned or sinusoidal")

    @property
    def feedforward_dim(self) -> int:
        return self.d_ff or 4 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
