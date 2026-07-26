from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .experts import AffineExperts, LowRankExperts, RoutesOnlyExperts, SharedBasisExperts, VectorExperts
from .gates import ZeroInitGate
from .initializers import initialize_keys
from .interface import MemoryLayer, RouteResult
from .routers import ChunkedExactTopKRouter, ExactTopKRouter, routing_diagnostics


@dataclass
class SparseMemoryConfig:
    d_model: int = 16
    num_supports: int = 32
    top_k: int = 4
    expert_mode: str = "vector"
    expert_rank: int = 4
    geometry_mode: str = "learned_full"
    metric: str = "dot"
    temperature: float = 1.0
    router_chunk_size: int | None = None
    include_global: bool = True
    gate_init: float = 0.0


class SparseSeparableMemory(MemoryLayer):
    """Top-k support memory with separable routing and local experts."""

    FIXED_GEOMETRIES = {"fixed_random", "fixed_data_sample", "fixed_kmeans", "fixed_farthest_point"}

    def __init__(self, config: SparseMemoryConfig | None = None, *, seed: int = 0, key_data: Tensor | None = None) -> None:
        super().__init__()
        self.config = config or SparseMemoryConfig()
        if self.config.num_supports <= 0 or self.config.top_k <= 0:
            raise ValueError("num_supports and top_k must be positive")
        if self.config.expert_mode not in {"vector", "affine", "low_rank", "shared_basis", "routes_only"}:
            raise ValueError("expert_mode must be vector, affine, low_rank, shared_basis, or routes_only")
        if self.config.geometry_mode not in self.FIXED_GEOMETRIES | {"learned_full", "learned_low_rank_delta", "product_key", "episodic_observed"}:
            raise ValueError("unknown geometry_mode")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if key_data is not None and self.config.geometry_mode in {"fixed_data_sample", "fixed_kmeans", "fixed_farthest_point"}:
            keys = initialize_keys(key_data, self.config.num_supports, self.config.geometry_mode, seed=seed).to(dtype=torch.float32)
        else:
            keys = torch.randn(self.config.num_supports, self.config.d_model, generator=generator)
        if keys.shape != (self.config.num_supports, self.config.d_model):
            raise ValueError("key_data must have feature dimension d_model")
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.keys = nn.Parameter(keys, requires_grad=self.config.geometry_mode not in self.FIXED_GEOMETRIES)
        if self.config.router_chunk_size:
            self.router = ChunkedExactTopKRouter(
                top_k=self.config.top_k,
                metric=self.config.metric,
                temperature=self.config.temperature,
                chunk_size=self.config.router_chunk_size,
            )
        else:
            self.router = ExactTopKRouter(
                top_k=self.config.top_k,
                metric=self.config.metric,
                temperature=self.config.temperature,
            )
        if self.config.expert_mode == "vector":
            self.experts: nn.Module = VectorExperts(self.config.num_supports, self.config.d_model)
        elif self.config.expert_mode == "affine":
            self.experts = AffineExperts(self.config.num_supports, self.config.d_model)
        elif self.config.expert_mode == "low_rank":
            self.experts = LowRankExperts(self.config.num_supports, self.config.d_model, self.config.expert_rank)
        elif self.config.expert_mode == "shared_basis":
            self.experts = SharedBasisExperts(self.config.num_supports, self.config.d_model, self.config.expert_rank)
        else:
            self.experts = RoutesOnlyExperts(self.config.d_model)
        self.global_value = nn.Parameter(torch.zeros(self.config.d_model)) if self.config.include_global else None
        self.gate = ZeroInitGate(self.config.gate_init)

    def route(self, query: Tensor, *, return_diagnostics: bool = False) -> RouteResult:
        del return_diagnostics
        flat = query.reshape(-1, query.shape[-1])
        route = self.router(flat, self.keys)
        route.diagnostics.update(routing_diagnostics(route, self.config.num_supports))
        return route

    def retrieve(self, query: Tensor, route: RouteResult) -> Tensor:
        flat = query.reshape(-1, query.shape[-1])
        update = self.experts(flat, route)
        if self.global_value is not None:
            update = update + self.global_value
        return update

    def forward(self, query: Tensor, return_diagnostics: bool = False):
        shape = query.shape
        if query.ndim < 2 or query.shape[-1] != self.config.d_model:
            raise ValueError("query must have shape [..., d_model]")
        flat = query.reshape(-1, query.shape[-1])
        route = self.route(flat)
        update = self.gate(self.retrieve(flat, route)).reshape(shape)
        if return_diagnostics:
            diagnostics = dict(route.diagnostics)
            diagnostics["gate_scale"] = float(self.gate.scale.detach())
            diagnostics["active_parameters_per_token"] = float(self.active_parameters_per_token)
            return update, diagnostics
        return update

    @property
    def active_parameters_per_token(self) -> int:
        return int(getattr(self.experts, "active_parameters_per_token", self.config.d_model) * min(self.config.top_k, self.config.num_supports))

    def parameter_accounting(self) -> dict[str, float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": float(total),
            "trainable_parameters": float(trainable),
            "active_parameters_per_token": float(self.active_parameters_per_token),
            "support_count": float(self.config.num_supports),
            "top_k": float(min(self.config.top_k, self.config.num_supports)),
        }

    @torch.no_grad()
    def update_algebra(self, **updates: Tensor) -> dict[str, Any]:
        applied: list[str] = []
        targets: dict[str, Tensor] = {"global_value": self.global_value} if self.global_value is not None else {}
        for name, value in updates.items():
            target = targets.get(name)
            if target is None and hasattr(self.experts, name):
                target = getattr(self.experts, name)
            if target is None:
                raise KeyError(f"unknown algebra parameter: {name}")
            if not isinstance(target, Tensor) or target.shape != value.shape:
                raise ValueError(f"shape mismatch for algebra parameter: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite algebra update: {name}")
            target.copy_(value)
            applied.append(name)
        return {"accepted": True, "applied": applied}

    @torch.no_grad()
    def update_geometry(
        self,
        candidate: Tensor,
        *,
        trust_radius: float = float("inf"),
        objective: Any | None = None,
        acceptance_margin: float = 0.0,
    ) -> dict[str, Any]:
        current = self.keys.detach().clone()
        if self.config.geometry_mode in self.FIXED_GEOMETRIES:
            return {"accepted": False, "reason": "geometry_fixed"}
        if candidate.shape != current.shape or not torch.isfinite(candidate).all():
            return {"accepted": False, "reason": "nonfinite_or_shape"}
        step_norm = float((candidate - current).norm())
        if step_norm > trust_radius:
            return {"accepted": False, "reason": "trust_region", "step_norm": step_norm}
        old_score = float(objective(current)) if objective is not None else None
        new_score = float(objective(candidate)) if objective is not None else None
        if old_score is not None and new_score > old_score + acceptance_margin:
            return {"accepted": False, "reason": "objective_increase", "step_norm": step_norm}
        self.keys.copy_(candidate)
        return {"accepted": True, "reason": "accepted", "step_norm": step_norm, "old_objective": old_score, "new_objective": new_score}
