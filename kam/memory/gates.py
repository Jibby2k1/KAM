from __future__ import annotations

import torch
from torch import Tensor, nn


class ZeroInitGate(nn.Module):
    """A scalar residual gate initialized to exactly zero."""

    def __init__(self, init: float = 0.0) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(float(init)))

    @property
    def scale(self) -> Tensor:
        return torch.tanh(self.logit)

    def forward(self, update: Tensor) -> Tensor:
        return self.scale.to(dtype=update.dtype, device=update.device) * update
