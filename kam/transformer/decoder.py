from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .config import TransformerConfig
from .layers import RMSNorm, SinusoidalPosition, SwiGLU


class DecoderBlock(nn.Module):
    """Pre-norm causal self-attention followed by a SwiGLU MLP."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model, config.norm_eps)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_ff = RMSNorm(config.d_model, config.norm_eps)
        self.ff = SwiGLU(config.d_model, config.feedforward_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        h = self.norm_attn(x)
        attended, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ff(self.norm_ff(x)))


class ModernDecoder(nn.Module):
    """A common decoder shell for T0 and every Phase 6 memory variant.

    ``memory_layers`` contains one module per decoder block.  A memory module
    returns a residual update, so a zero-initialized memory gate is exactly a
    baseline-equivalence control at initialization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        memory_layers: Iterable[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("ModernDecoder requires vocab_size for token input")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.positional_encoding == "learned":
            self.position = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.position = SinusoidalPosition(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        supplied = list(memory_layers or [])
        if supplied and len(supplied) != config.n_layers:
            raise ValueError("memory_layers must contain one module per decoder block")
        self.memory_layers = nn.ModuleList(supplied)
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _add_position(self, x: Tensor) -> Tensor:
        if isinstance(self.position, nn.Embedding):
            if x.shape[1] > self.config.max_seq_len:
                raise ValueError("sequence length exceeds positional capacity")
            positions = torch.arange(x.shape[1], device=x.device)
            return x + self.position(positions).unsqueeze(0)
        return self.position(x)

    @property
    def active_parameters_per_token(self) -> int:
        """Count dense parameters plus the declared active memory path.

        Dense decoder parameters are active for every token. Memory controls
        may expose ``active_parameters_per_token`` for their routed path; this
        keeps Stage 2's resource view separate from total stored parameters.
        """
        memory_parameter_ids = {
            id(parameter)
            for layer in self.memory_layers
            for parameter in layer.parameters()
        }
        dense = sum(parameter.numel() for parameter in self.parameters() if id(parameter) not in memory_parameter_ids)
        active_memory = sum(
            int(getattr(layer, "active_parameters_per_token", sum(parameter.numel() for parameter in layer.parameters())))
            for layer in self.memory_layers
        )
        return int(dense + active_memory)

    def forward(self, input_ids: Tensor, return_diagnostics: bool = False):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        x = self._add_position(self.token_embedding(input_ids))
        steps = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        diagnostics: list[dict[str, float]] = []
        for index, block in enumerate(self.blocks):
            x = block(x, causal_mask)
            if self.memory_layers:
                result = self.memory_layers[index](x, return_diagnostics=return_diagnostics)
                if return_diagnostics:
                    update, info = result
                    diagnostics.append(info)
                else:
                    update = result
                x = x + update
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        if return_diagnostics:
            return logits, diagnostics
        return logits
