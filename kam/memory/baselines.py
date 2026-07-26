from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .interface import MemoryLayer, RouteResult
from .routers import ExactTopKRouter, ProductKeyRouter, routing_diagnostics


class MemoryTokenLayer(MemoryLayer):
    """Learnable memory-token control (T-MEMTOK)."""

    def __init__(self, d_model: int, num_tokens: int = 16, top_k: int = 4, gate_init: float = 0.0) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.router = ExactTopKRouter(top_k=top_k, metric="dot")
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def route(self, query: Tensor) -> RouteResult:
        route = self.router(query.reshape(-1, query.shape[-1]), self.tokens)
        route.diagnostics.update(routing_diagnostics(route, self.tokens.shape[0]))
        return route

    def retrieve(self, query: Tensor, routing: RouteResult) -> Tensor:
        del query
        return (routing.weights.unsqueeze(-1) * self.tokens[routing.indices]).sum(1)

    @property
    def active_parameters_per_token(self) -> int:
        return int(min(self.router.top_k, self.tokens.shape[0]) * self.tokens.shape[-1])

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        route = self.route(query)
        value = torch.tanh(self.gate) * self.retrieve(query.reshape(-1, query.shape[-1]), route)
        value = value.reshape_as(query)
        if return_diagnostics:
            info = dict(route.diagnostics)
            info["gate_scale"] = float(torch.tanh(self.gate).detach())
            return value, info
        return value

    def update_algebra(self, **updates):
        if "tokens" not in updates or updates["tokens"].shape != self.tokens.shape:
            raise ValueError("memory-token algebra update must provide tokens with matching shape")
        with torch.no_grad():
            self.tokens.copy_(updates["tokens"])
        return {"accepted": True, "applied": ["tokens"]}

    def update_geometry(self, candidate: Tensor, **kwargs):
        del kwargs
        return {"accepted": False, "reason": "memory_tokens_have_no_separate_geometry"}

    def diagnostics(self, route: RouteResult | None = None):
        return dict(route.diagnostics) if route is not None else {}


class MixtureOfExpertsMemory(nn.Module):
    """Small top-k expert residual used as a matched-compute MoE control."""

    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, d_hidden: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        hidden = d_hidden or 4 * d_model
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)) for _ in range(num_experts)])

    @property
    def active_parameters_per_token(self) -> int:
        return sum(parameter.numel() for parameter in self.experts[0].parameters()) * self.top_k

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        flat = query.reshape(-1, query.shape[-1])
        scores = self.router(flat)
        values, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        expert_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        selected = expert_outputs.gather(1, indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
        output = (weights.unsqueeze(-1) * selected).sum(1).reshape_as(query)
        if return_diagnostics:
            counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).float()
            info = {
                "routing_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum(-1)).mean()),
                "active_support_fraction": float((counts > 0).float().mean()),
                "load_balance_error": float(((counts / counts.sum().clamp_min(1e-12) - 1 / self.num_experts) ** 2).mean()),
            }
            return output, info
        return output


class ProductKeyMemory(nn.Module):
    """Product-key memory control with factorized codebooks."""

    def __init__(self, d_model: int, codebook_size: int = 16, top_k: int = 4) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError("product-key d_model must be even")
        half = d_model // 2
        self.codebook_a = nn.Parameter(torch.randn(codebook_size, half) * 0.02)
        self.codebook_b = nn.Parameter(torch.randn(codebook_size, half) * 0.02)
        self.values = nn.Parameter(torch.randn(codebook_size * codebook_size, d_model) * 0.02)
        self.router = ProductKeyRouter(top_k=top_k)

    @property
    def active_parameters_per_token(self) -> int:
        return self.router.top_k * self.values.shape[-1]

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        shape = query.shape
        flat = query.reshape(-1, shape[-1])
        route = self.router(flat, self.codebook_a, self.codebook_b)
        output = (route.weights.unsqueeze(-1) * self.values[route.indices]).sum(1).reshape(shape)
        if return_diagnostics:
            info = routing_diagnostics(route, self.values.shape[0])
            return output, info
        return output


class DualMemory(nn.Module):
    """Global persistent plus observed episodic residual control."""

    def __init__(self, persistent: nn.Module, episodic: nn.Module) -> None:
        super().__init__()
        self.persistent = persistent
        self.episodic = episodic

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        persistent = self.persistent(query, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            persistent_value, persistent_info = persistent
        else:
            persistent_value, persistent_info = persistent, {}
        try:
            episodic_value = self.episodic(query)
        except RuntimeError:
            episodic_value = torch.zeros_like(query)
        output = persistent_value + episodic_value
        if return_diagnostics:
            persistent_info["episodic_active"] = float(bool(torch.any(episodic_value.detach())))
            return output, persistent_info
        return output


__all__ = ["DualMemory", "MemoryTokenLayer", "MixtureOfExpertsMemory", "ProductKeyMemory"]
