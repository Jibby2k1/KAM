from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from .attention import (
    BandwidthMode,
    KernelMemoryAttention,
    KernelSelfAttention,
    RadialMetric,
    ScoreType,
)

TaskType = Literal["language", "regression"]
PoolingType = Literal["last", "mean"]
MemoryOutput = Literal["residual", "routes", "both"]
RouteFeatures = Literal["raw", "projected"]


@dataclass
class ModelDiagnostics:
    context_weights: list[Tensor]
    memory_weights: list[Tensor]
    memory_features: list[Tensor] = field(default_factory=list)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = expansion * d_model
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class KAMBlock(nn.Module):
    """A block with independently configurable context and memory geometries."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_supports: int,
        score_type: ScoreType = "rbf",
        context_window: int | None = None,
        dropout: float = 0.0,
        use_context: bool = True,
        use_memory: bool = True,
        context_score: ScoreType | None = None,
        memory_score: ScoreType | None = None,
        context_normalize_qk: bool | None = None,
        memory_normalize_qk: bool | None = None,
        radial_metric: RadialMetric = "diagonal",
        bandwidth: BandwidthMode = "learned",
        init_bandwidth: float = 1.0,
        memory_output: MemoryOutput = "residual",
        route_features: RouteFeatures = "raw",
        route_projection_dim: int | None = None,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        if memory_output not in {"residual", "routes", "both"}:
            raise ValueError(f"Unsupported memory output: {memory_output}")
        if route_features not in {"raw", "projected"}:
            raise ValueError(f"Unsupported route feature mode: {route_features}")
        self.use_context = use_context
        self.use_memory = use_memory
        self.memory_output = memory_output
        self.route_enabled = use_memory and memory_output in {"routes", "both"}
        context_score = context_score or score_type
        memory_score = memory_score or score_type
        if context_normalize_qk is None:
            context_normalize_qk = context_score in {"rbf", "radial", "cosine"}
        if memory_normalize_qk is None:
            memory_normalize_qk = memory_score in {"rbf", "radial", "cosine"}

        self.context_norm = nn.LayerNorm(d_model)
        self.context = (
            KernelSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                score_type=context_score,
                window=context_window,
                dropout=dropout,
                normalize_qk=context_normalize_qk,
                radial_metric=radial_metric,
                bandwidth=bandwidth,
                init_bandwidth=init_bandwidth,
            )
            if use_context
            else None
        )
        self.memory_norm = nn.LayerNorm(d_model)
        self.memory = (
            KernelMemoryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_supports=num_supports,
                score_type=memory_score,
                dropout=dropout,
                normalize_qk=memory_normalize_qk,
                radial_metric=radial_metric,
                bandwidth=bandwidth,
                init_bandwidth=init_bandwidth,
            )
            if use_memory
            else None
        )
        route_input_dim = num_heads * num_supports
        self.route_output_dim = route_input_dim
        self.route_projection = None
        if self.route_enabled and route_features == "projected":
            self.route_output_dim = int(route_projection_dim or d_model)
            self.route_projection = nn.Linear(route_input_dim, self.route_output_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, expansion=ffn_expansion, dropout=dropout)
        self.last_route_features: Tensor | None = None

    def forward(
        self, inputs: Tensor, return_weights: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        context_weights = None
        memory_weights = None
        self.last_route_features = None
        hidden = inputs
        if self.context is not None:
            update, context_weights = self.context(self.context_norm(hidden), return_weights)
            hidden = hidden + update
        if self.memory is not None:
            need_memory_weights = return_weights or self.route_enabled
            update, memory_weights = self.memory(
                self.memory_norm(hidden), need_memory_weights
            )
            if self.memory_output in {"residual", "both"}:
                hidden = hidden + update
            if self.route_enabled and memory_weights is not None:
                routes = memory_weights.permute(0, 2, 1, 3).reshape(
                    memory_weights.shape[0], memory_weights.shape[2], -1
                )
                self.last_route_features = (
                    self.route_projection(routes) if self.route_projection is not None else routes
                )
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, context_weights, memory_weights


class KAMSequenceModel(nn.Module):
    """Compact KAM model with legacy and Phase II configuration paths."""

    def __init__(
        self,
        task: TaskType,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_supports: int = 64,
        score_type: ScoreType = "rbf",
        context_window: int | None = None,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        vocab_size: int | None = None,
        input_dim: int | None = None,
        output_dim: int = 1,
        use_context: bool = True,
        use_memory: bool = True,
        regression_pool: PoolingType = "last",
        expose_memory_weights: bool = True,
        context_score: ScoreType | None = None,
        memory_score: ScoreType | None = None,
        context_normalize_qk: bool | None = None,
        memory_normalize_qk: bool | None = None,
        radial_metric: RadialMetric = "diagonal",
        bandwidth: BandwidthMode = "learned",
        init_bandwidth: float = 1.0,
        memory_output: MemoryOutput = "residual",
        route_features: RouteFeatures = "raw",
        route_projection_dim: int | None = None,
        position_mode: Literal["learned", "sinusoidal"] = "learned",
        ffn_expansion: int = 4,
        parameter_match_target: int | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_supports = num_supports
        self.use_context = use_context
        self.use_memory = use_memory
        self.regression_pool = regression_pool
        self.expose_memory_weights = expose_memory_weights and use_memory
        if position_mode not in {"learned", "sinusoidal"}:
            raise ValueError(f"Unsupported position mode: {position_mode}")
        self.position_mode = position_mode
        self.max_seq_len = max_seq_len
        self.context_score = context_score or score_type
        self.memory_score = memory_score or score_type
        self.memory_output = memory_output
        self.route_features = route_features
        self.route_projection_dim = route_projection_dim

        if task == "language":
            if vocab_size is None:
                raise ValueError("vocab_size is required for language modeling.")
            self.input_layer: nn.Module = nn.Embedding(vocab_size, d_model)
            self.output_size = vocab_size
        elif task == "regression":
            if input_dim is None:
                raise ValueError("input_dim is required for regression.")
            self.input_layer = nn.Linear(input_dim, d_model)
            self.output_size = output_dim
        else:
            raise ValueError(f"Unsupported task: {task}")

        if position_mode == "learned":
            self.position = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.position, std=0.02)
        else:
            self.register_buffer("position", torch.empty(0), persistent=False)
        self.blocks = nn.ModuleList(
            [
                KAMBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_supports=num_supports,
                    score_type=score_type,
                    context_score=context_score,
                    memory_score=memory_score,
                    context_normalize_qk=context_normalize_qk,
                    memory_normalize_qk=memory_normalize_qk,
                    radial_metric=radial_metric,
                    bandwidth=bandwidth,
                    init_bandwidth=init_bandwidth,
                    context_window=context_window,
                    dropout=dropout,
                    use_context=use_context,
                    use_memory=use_memory,
                    memory_output=memory_output,
                    route_features=route_features,
                    route_projection_dim=route_projection_dim,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.baseline_route_projection = None
        if not use_memory and route_features == "projected" and route_projection_dim is not None:
            self.baseline_route_projection = nn.Linear(d_model, int(route_projection_dim), bias=False)

        route_dim = 0
        if use_memory and memory_output in {"routes", "both"}:
            route_dim = self.blocks[-1].route_output_dim
        elif not use_memory and self.baseline_route_projection is not None:
            route_dim = int(route_projection_dim)
        elif task == "regression" and self.expose_memory_weights:
            # Legacy regression checkpoints expose the final raw route matrix.
            route_dim = num_heads * num_supports
        self.route_feature_dim = route_dim
        readout_input_dim = d_model + route_dim
        if task == "language":
            self.readout = nn.Linear(readout_input_dim, self.output_size)
            self.regression_feature_dim = None
        else:
            self.regression_feature_dim = readout_input_dim
            self.readout = nn.Linear(self.regression_feature_dim, self.output_size)
        self._add_capacity_padding(parameter_match_target)

    def _add_capacity_padding(self, target: int | None) -> None:
        if target is None:
            return
        current = sum(parameter.numel() for parameter in self.parameters())
        if current > int(target):
            raise ValueError(f"parameter_match_target={target} is below model size {current}")
        self.capacity_padding = nn.Parameter(torch.zeros(int(target) - current))


    def _position_encoding(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.position_mode == "learned":
            if length > self.max_seq_len:
                raise ValueError(f"Sequence length {length} exceeds max_seq_len={self.max_seq_len} for learned positions.")
            return self.position[:, :length, :].to(dtype=dtype)
        positions = torch.arange(length, device=device, dtype=dtype)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.d_model))
        encoding = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions[:, None] * div_term)
        encoding[:, 1::2] = torch.cos(positions[:, None] * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def encode(self, inputs: Tensor, return_weights: bool = False) -> tuple[Tensor, ModelDiagnostics]:
        hidden = self.input_layer(inputs)
        hidden = hidden + self._position_encoding(hidden.shape[1], hidden.device, hidden.dtype)
        context_weights: list[Tensor] = []
        memory_weights: list[Tensor] = []
        memory_features: list[Tensor] = []
        for block in self.blocks:
            hidden, context, memory = block(hidden, return_weights=return_weights)
            if return_weights and context is not None:
                context_weights.append(context)
            if return_weights and memory is not None:
                memory_weights.append(memory)
            if block.last_route_features is not None:
                memory_features.append(block.last_route_features)
        hidden = self.final_norm(hidden)
        return hidden, ModelDiagnostics(context_weights, memory_weights, memory_features)

    def _append_route_features(
        self, pooled: Tensor, diagnostics: ModelDiagnostics
    ) -> Tensor:
        if diagnostics.memory_features:
            route = diagnostics.memory_features[-1]
            if self.regression_pool == "last":
                route = route[:, -1, :]
            else:
                route = route.mean(dim=1)
            return torch.cat([pooled, route], dim=-1)
        if self.expose_memory_weights and diagnostics.memory_weights:
            final_memory = diagnostics.memory_weights[-1]
            if self.regression_pool == "last":
                memory_features = final_memory[:, :, -1, :]
            else:
                memory_features = final_memory.mean(dim=2)
            return torch.cat([pooled, memory_features.flatten(start_dim=1)], dim=-1)
        if self.baseline_route_projection is not None:
            return torch.cat([pooled, self.baseline_route_projection(pooled)], dim=-1)
        return pooled

    def regression_features(
        self, inputs: Tensor, return_weights: bool = False
    ) -> tuple[Tensor, ModelDiagnostics]:
        if self.task != "regression":
            raise RuntimeError("regression_features is available only for regression models.")
        hidden, diagnostics = self.encode(inputs, return_weights=True)
        if self.regression_pool == "last":
            pooled = hidden[:, -1, :]
        elif self.regression_pool == "mean":
            pooled = hidden.mean(dim=1)
        else:
            raise RuntimeError(f"Unsupported pooling mode: {self.regression_pool}")
        pooled = self._append_route_features(pooled, diagnostics)
        if not return_weights:
            diagnostics = ModelDiagnostics([], [], diagnostics.memory_features)
        return pooled, diagnostics

    def forward(
        self, inputs: Tensor, return_weights: bool = False
    ) -> tuple[Tensor, ModelDiagnostics] | Tensor:
        if self.task == "language":
            need_routes = self.memory_output in {"routes", "both"}
            hidden, diagnostics = self.encode(
                inputs, return_weights=return_weights or need_routes
            )
            if diagnostics.memory_features:
                route = diagnostics.memory_features[-1]
                hidden = torch.cat([hidden, route], dim=-1)
            logits = self.readout(hidden)
            return (logits, diagnostics) if return_weights else logits

        features, diagnostics = self.regression_features(inputs, return_weights=return_weights)
        predictions = self.readout(features)
        return (predictions, diagnostics) if return_weights else predictions
