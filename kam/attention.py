from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ScoreType = Literal["rbf", "radial", "dot", "cosine"]
RadialMetric = Literal["isotropic", "diagonal"]
BandwidthMode = Literal["fixed", "learned"]


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("Softplus targets must be positive.")
    return math.log(math.expm1(value))


class PairwiseAttentionScore(nn.Module):
    """Compute dot, cosine, or radial attention scores.

    The legacy name ``rbf`` is retained as an alias for ``radial`` so old
    checkpoints keep loading. Radial attention uses a positive diagonal or
    isotropic metric and exposes both a direct-distance reference and the
    expanded bilinear-plus-key-bias form used by the fast path.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        score_type: ScoreType = "rbf",
        normalize_qk: bool = True,
        init_bandwidth: float = 1.0,
        min_bandwidth: float = 1e-3,
        radial_metric: RadialMetric = "diagonal",
        bandwidth: BandwidthMode = "learned",
    ) -> None:
        super().__init__()
        if radial_metric not in {"isotropic", "diagonal"}:
            raise ValueError(f"Unsupported radial metric: {radial_metric}")
        if bandwidth not in {"fixed", "learned"}:
            raise ValueError(f"Unsupported bandwidth mode: {bandwidth}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.score_type = "rbf" if score_type == "radial" else score_type
        self.normalize_qk = normalize_qk or self.score_type == "cosine"
        self.min_bandwidth = min_bandwidth
        self.radial_metric = radial_metric
        self.bandwidth_mode = bandwidth

        if self.score_type not in {"rbf", "dot", "cosine"}:
            raise ValueError(f"Unsupported score type: {score_type}")
        if self.score_type == "rbf":
            metric_shape = (num_heads, 1) if radial_metric == "isotropic" else (num_heads, head_dim)
            self.raw_metric = nn.Parameter(torch.full(metric_shape, _inverse_softplus(1.0)))
            if bandwidth == "learned":
                target = max(init_bandwidth - min_bandwidth, 1e-4)
                self.raw_bandwidth = nn.Parameter(
                    torch.full((num_heads,), _inverse_softplus(target))
                )
            else:
                self.register_buffer(
                    "fixed_bandwidth",
                    torch.full((num_heads,), max(init_bandwidth, min_bandwidth)),
                )

    def _prepare(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        if queries.ndim != 4 or keys.ndim != 4:
            raise ValueError("queries and keys must have shape [B, H, T, R].")
        if queries.shape[:2] != keys.shape[:2] or queries.shape[-1] != keys.shape[-1]:
            raise ValueError("queries and keys must agree in batch, head, and feature dimensions.")
        if self.normalize_qk:
            queries = F.normalize(queries, dim=-1)
            keys = F.normalize(keys, dim=-1)
        return queries, keys

    def metric(self) -> Tensor:
        if self.score_type != "rbf":
            raise RuntimeError("The non-radial scores have no radial metric.")
        metric = F.softplus(self.raw_metric) + 1e-6
        if self.radial_metric == "isotropic":
            metric = metric.expand(self.num_heads, self.head_dim)
        return metric

    def bandwidth(self) -> Tensor:
        if self.score_type != "rbf":
            raise RuntimeError("The non-radial scores have no bandwidth.")
        if self.bandwidth_mode == "fixed":
            return self.fixed_bandwidth
        return F.softplus(self.raw_bandwidth) + self.min_bandwidth

    def direct_radial_scores(self, queries: Tensor, keys: Tensor) -> Tensor:
        """Reference implementation using explicit pairwise distances."""
        queries, keys = self._prepare(queries, keys)
        metric = self.metric().sqrt()[None, :, None, :]
        q_metric = queries * metric
        k_metric = keys * metric
        q_norm = (q_metric * q_metric).sum(dim=-1, keepdim=True)
        k_norm = (k_metric * k_metric).sum(dim=-1).unsqueeze(-2)
        cross = torch.matmul(q_metric, k_metric.transpose(-1, -2))
        distance_sq = (q_norm + k_norm - 2.0 * cross).clamp_min(0.0)
        bandwidth_sq = self.bandwidth()[None, :, None, None].square()
        return -0.5 * distance_sq / bandwidth_sq

    def expanded_radial_scores(
        self,
        queries: Tensor,
        keys: Tensor,
        *,
        include_query_term: bool = True,
    ) -> Tensor:
        """Expanded radial score; query-only energy may be omitted before softmax."""
        queries, keys = self._prepare(queries, keys)
        metric = self.metric()[None, :, None, :]
        q_norm = (queries.square() * metric).sum(dim=-1, keepdim=True)
        k_norm = (keys.square() * metric).sum(dim=-1).unsqueeze(-2)
        cross = torch.matmul(queries * metric, keys.transpose(-1, -2))
        numerator = 2.0 * cross - k_norm
        if include_query_term:
            numerator = numerator - q_norm
        bandwidth_sq = self.bandwidth()[None, :, None, None].square()
        return 0.5 * numerator / bandwidth_sq

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        queries, keys = self._prepare(queries, keys)
        if self.score_type == "dot":
            return torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.score_type == "cosine":
            return torch.matmul(queries, keys.transpose(-1, -2))
        # The query-only term cancels under row-wise softmax.
        return self.expanded_radial_scores(queries, keys, include_query_term=False)


class RelativeLagBias(nn.Module):
    """Learn one scalar bias per attention head and causal lag."""

    def __init__(self, num_heads: int, max_lag: int) -> None:
        super().__init__()
        if max_lag < 1:
            raise ValueError("max_lag must be positive.")
        self.max_lag = max_lag
        self.bias = nn.Parameter(torch.zeros(num_heads, max_lag))

    def forward(self, length: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device)
        lag = positions[:, None] - positions[None, :]
        lag = lag.clamp(min=0, max=self.max_lag - 1)
        return self.bias[:, lag]


class KernelSelfAttention(nn.Module):
    """Multi-head causal self-attention with configurable score geometry."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        score_type: ScoreType = "rbf",
        window: int | None = None,
        dropout: float = 0.0,
        normalize_qk: bool = True,
        use_relative_bias: bool = True,
        max_relative_lag: int = 512,
        radial_metric: RadialMetric = "diagonal",
        bandwidth: BandwidthMode = "learned",
        init_bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.score = PairwiseAttentionScore(
            num_heads=num_heads,
            head_dim=self.head_dim,
            score_type=score_type,
            normalize_qk=normalize_qk,
            radial_metric=radial_metric,
            bandwidth=bandwidth,
            init_bandwidth=init_bandwidth,
        )
        relative_limit = window if window is not None else max_relative_lag
        self.relative_bias = (
            RelativeLagBias(num_heads, relative_limit) if use_relative_bias else None
        )
        self.dropout = nn.Dropout(dropout)

    def _allowed_mask(self, length: int, device: torch.device) -> Tensor:
        rows = torch.arange(length, device=device)[:, None]
        cols = torch.arange(length, device=device)[None, :]
        lag = rows - cols
        allowed = lag >= 0
        if self.window is not None:
            allowed = allowed & (lag < self.window)
        return allowed

    def forward(self, inputs: Tensor, return_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = inputs.shape
        qkv = self.qkv(inputs).view(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)
        scores = self.score(queries, keys)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias(length, inputs.device)[None, :, :, :]
        allowed = self._allowed_mask(length, inputs.device)
        scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, values)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        output = self.output(context)
        return output, weights if return_weights else None


class KernelMemoryAttention(nn.Module):
    """Cross-attention from tokens to a fixed-size learned support bank."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_supports: int,
        score_type: ScoreType = "rbf",
        dropout: float = 0.0,
        normalize_qk: bool = True,
        radial_metric: RadialMetric = "diagonal",
        bandwidth: BandwidthMode = "learned",
        init_bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_supports < 1:
            raise ValueError("num_supports must be positive.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_supports = num_supports
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.memory_keys = nn.Parameter(
            torch.randn(num_heads, num_supports, self.head_dim) / math.sqrt(self.head_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(num_heads, num_supports, self.head_dim) / math.sqrt(self.head_dim)
        )
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.score = PairwiseAttentionScore(
            num_heads=num_heads,
            head_dim=self.head_dim,
            score_type=score_type,
            normalize_qk=normalize_qk,
            radial_metric=radial_metric,
            bandwidth=bandwidth,
            init_bandwidth=init_bandwidth,
        )
        self.dropout = nn.Dropout(dropout)
        self.support_mask: Tensor | None = None
    def set_support_mask(self, mask: Tensor | None) -> None:
        if mask is not None and (mask.ndim != 1 or mask.numel() != self.num_supports):
            raise ValueError("support mask must have shape [num_supports].")
        self.support_mask = None if mask is None else mask.detach().to(dtype=torch.bool, device=self.memory_keys.device)


    def forward(self, inputs: Tensor, return_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = inputs.shape
        queries = self.query(inputs).view(batch, length, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)
        keys = self.memory_keys[None, :, :, :].expand(batch, -1, -1, -1)
        values = self.memory_values[None, :, :, :].expand(batch, -1, -1, -1)
        scores = self.score(queries, keys)
        if self.support_mask is not None:
            scores = scores.masked_fill(~self.support_mask[None, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, values)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        output = self.output(context)
        return output, weights if return_weights else None
