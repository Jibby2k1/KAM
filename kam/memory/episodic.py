from __future__ import annotations

import torch
from torch import Tensor, nn

from .routers import ExactTopKRouter


class EpisodicMemory(nn.Module):
    """A bounded observed-key/value bank for the online lane.

    Writes are explicit and non-differentiable.  This gives Stage 0 a small
    global-versus-episodic interface without silently changing optimizer
    semantics in the learned persistent bank.
    """

    def __init__(self, capacity: int, d_model: int, top_k: int = 1) -> None:
        super().__init__()
        if capacity <= 0 or d_model <= 0:
            raise ValueError("capacity and d_model must be positive")
        self.capacity = capacity
        self.d_model = d_model
        self.register_buffer("keys", torch.zeros(capacity, d_model))
        self.register_buffer("values", torch.zeros(capacity, d_model))
        self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("write_index", torch.zeros((), dtype=torch.long))
        self.router = ExactTopKRouter(top_k=top_k, metric="negative_l2")

    @torch.no_grad()
    def write(self, keys: Tensor, values: Tensor) -> None:
        if keys.shape != values.shape or keys.ndim != 2 or keys.shape[-1] != self.d_model:
            raise ValueError("keys and values must have shape [batch, d_model]")
        for key, value in zip(keys, values):
            index = int(self.write_index.item()) % self.capacity
            self.keys[index].copy_(key)
            self.values[index].copy_(value)
            self.valid[index] = True
            self.write_index.add_(1)

    def route(self, query: Tensor):
        valid = self.valid
        if not bool(valid.any()):
            raise RuntimeError("episodic memory is empty")
        return self.router(query, self.keys[valid])

    def forward(self, query: Tensor) -> Tensor:
        flat = query.reshape(-1, query.shape[-1])
        route = self.route(flat)
        return (route.weights.unsqueeze(-1) * self.values[self.valid][route.indices]).sum(1).reshape_as(query)
