from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMSNorm with an explicit epsilon for stable tiny fixtures."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * scale.to(dtype=x.dtype)) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block used by the modern decoder control."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(gate) * value))


class SinusoidalPosition(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_seq_len, d_model)
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.table.shape[0]:
            raise ValueError("sequence length exceeds positional capacity")
        return x + self.table[: x.shape[1]].to(device=x.device, dtype=x.dtype).unsqueeze(0)
