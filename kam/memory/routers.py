from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .interface import RouteResult

Metric = Literal["dot", "negative_l2", "cosine"]


def _flatten_query(query: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    if query.ndim < 2:
        raise ValueError("query must have shape [..., d]")
    return query.reshape(-1, query.shape[-1]), tuple(query.shape[:-1])


def pairwise_scores(query: Tensor, keys: Tensor, metric: Metric = "dot") -> Tensor:
    q, _ = _flatten_query(query)
    if keys.ndim != 2 or q.shape[-1] != keys.shape[-1]:
        raise ValueError("query and keys must have matching feature dimensions")
    if metric == "dot":
        return q @ keys.T
    if metric == "negative_l2":
        return -(q.square().sum(-1, keepdim=True) + keys.square().sum(-1).unsqueeze(0) - 2 * q @ keys.T)
    if metric == "cosine":
        return torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(keys, dim=-1).T
    raise ValueError(f"unknown routing metric: {metric}")


def _make_route(scores: Tensor, top_k: int, temperature: float, num_supports: int) -> RouteResult:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [tokens, supports]")
    k = min(int(top_k), scores.shape[-1])
    if k < 1:
        raise ValueError("top_k must be positive")
    values, indices = torch.topk(scores, k=k, dim=-1)
    weights = torch.softmax(values / max(float(temperature), 1e-8), dim=-1)
    route = RouteResult(indices=indices.long(), weights=weights, scores=values, num_supports=num_supports)
    route.validate(tokens=scores.shape[0], top_k=k)
    return route


class ExactTopKRouter(nn.Module):
    def __init__(self, top_k: int = 4, metric: Metric = "dot", temperature: float = 1.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.metric = metric
        self.temperature = temperature

    def forward(self, query: Tensor, keys: Tensor) -> RouteResult:
        scores = pairwise_scores(query, keys, self.metric)
        return _make_route(scores, self.top_k, self.temperature, keys.shape[0])

    route = forward


class ChunkedExactTopKRouter(ExactTopKRouter):
    """Exact top-k routing with bounded support-score working memory."""

    def __init__(self, top_k: int = 4, metric: Metric = "dot", temperature: float = 1.0, chunk_size: int = 1024) -> None:
        super().__init__(top_k=top_k, metric=metric, temperature=temperature)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def forward(self, query: Tensor, keys: Tensor) -> RouteResult:
        q, _ = _flatten_query(query)
        if keys.ndim != 2 or q.shape[-1] != keys.shape[-1]:
            raise ValueError("query and keys must have matching feature dimensions")
        best_scores: Tensor | None = None
        best_indices: Tensor | None = None
        for start in range(0, keys.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, keys.shape[0])
            scores = pairwise_scores(q, keys[start:stop], self.metric)
            local_k = min(self.top_k, scores.shape[-1])
            values, indices = torch.topk(scores, k=local_k, dim=-1)
            indices = indices + start
            if best_scores is None:
                best_scores, best_indices = values, indices
            else:
                candidates_scores = torch.cat((best_scores, values), dim=-1)
                candidates_indices = torch.cat((best_indices, indices), dim=-1)
                keep = min(self.top_k, candidates_scores.shape[-1])
                best_scores, positions = torch.topk(candidates_scores, keep, dim=-1)
                best_indices = candidates_indices.gather(-1, positions)
        assert best_scores is not None and best_indices is not None
        weights = torch.softmax(best_scores / max(float(self.temperature), 1e-8), dim=-1)
        route = RouteResult(best_indices.long(), weights, best_scores, keys.shape[0])
        route.validate(tokens=q.shape[0], top_k=min(self.top_k, keys.shape[0]))
        return route


class ProductKeyRouter(nn.Module):
    """Factorized product-key router with an exact materialized reference."""

    def __init__(self, top_k: int = 4, temperature: float = 1.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, query: Tensor, codebook_a: Tensor, codebook_b: Tensor) -> RouteResult:
        q, _ = _flatten_query(query)
        if q.shape[-1] != codebook_a.shape[-1] + codebook_b.shape[-1]:
            raise ValueError("query dimension must equal the two codebook dimensions")
        qa, qb = q.split((codebook_a.shape[-1], codebook_b.shape[-1]), dim=-1)
        scores_a = qa @ codebook_a.T
        scores_b = qb @ codebook_b.T
        scores = (scores_a.unsqueeze(-1) + scores_b.unsqueeze(-2)).reshape(q.shape[0], -1)
        return _make_route(scores, self.top_k, self.temperature, scores.shape[-1])


class ApproximateTopKRouter(ExactTopKRouter):
    """Optional approximate router using a bounded random-projection shortlist.

    Stage 3 compares this shortlist against ``ExactTopKRouter`` explicitly;
    approximation is never silently treated as exact.
    """

    def __init__(self, top_k: int = 4, metric: Metric = "dot", temperature: float = 1.0, candidate_size: int = 128, seed: int = 0) -> None:
        super().__init__(top_k=top_k, metric=metric, temperature=temperature)
        if candidate_size <= 0:
            raise ValueError("candidate_size must be positive")
        self.candidate_size = candidate_size
        self.seed = seed

    def forward(self, query: Tensor, keys: Tensor) -> RouteResult:
        q, _ = _flatten_query(query)
        if keys.ndim != 2 or q.shape[-1] != keys.shape[-1]:
            raise ValueError("query and keys must have matching feature dimensions")
        generator = torch.Generator(device=keys.device).manual_seed(self.seed)
        projection = torch.randn(keys.shape[-1], device=keys.device, dtype=keys.dtype, generator=generator)
        key_order = (keys @ projection).argsort()
        shortlist = key_order[: min(self.candidate_size, keys.shape[0])]
        candidate_scores = pairwise_scores(q, keys[shortlist], self.metric)
        route = _make_route(candidate_scores, self.top_k, self.temperature, keys.shape[0])
        route.indices = shortlist[route.indices]
        route.validate(tokens=q.shape[0], top_k=min(self.top_k, shortlist.numel()))
        return route


def routing_diagnostics(route: RouteResult, num_supports: int | None = None, dead_threshold: float = 1e-8) -> dict[str, float]:
    """Return support recall/entropy/load diagnostics from a sparse route."""
    num_supports = int(num_supports or route.num_supports)
    weights = route.weights.detach().float()
    counts = torch.zeros(num_supports, device=weights.device, dtype=weights.dtype)
    counts.scatter_add_(0, route.indices.reshape(-1), weights.reshape(-1))
    probabilities = counts / counts.sum().clamp_min(1e-12)
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(-1).mean()
    effective = 1.0 / weights.square().sum(-1).clamp_min(1e-12)
    used = torch.zeros(num_supports, device=weights.device, dtype=torch.bool)
    used[route.indices.reshape(-1)] = True
    target = counts.sum() / max(num_supports, 1)
    load_balance = (counts - target).square().mean() / target.square().mean().clamp_min(1e-12)
    unique_counts = torch.tensor([row.unique().numel() for row in route.indices], device=route.indices.device, dtype=weights.dtype)
    duplicate_fraction = 1.0 - unique_counts.mean() / max(route.indices.shape[-1], 1)
    return {
        "routing_entropy": float(entropy.detach()),
        "effective_support_count": float(effective.mean().detach()),
        "global_effective_support_count": float(1.0 / probabilities.square().sum().clamp_min(1e-12)),
        "dead_support_fraction": float((counts <= dead_threshold).float().mean().detach()),
        "load_balance_error": float(load_balance.detach()),
        "load_balance": float(load_balance.detach()),
        "duplicate_fraction": float(duplicate_fraction),
        "tokens_per_support": float(route.indices.numel() / max(num_supports, 1)),
        "tokens_per_used_support": float(route.indices.shape[0] / max(int(used.sum()), 1)),
        "active_support_fraction": float(used.float().mean().detach()),
    }


def recall_at_k(predicted: RouteResult, reference: RouteResult) -> float:
    if predicted.indices.shape[0] != reference.indices.shape[0]:
        raise ValueError("routes must contain the same number of tokens")
    pred = predicted.indices
    ref = reference.indices
    matches = (pred.unsqueeze(-1) == ref.unsqueeze(-2)).any(dim=-1).float()
    return float(matches.mean().detach())
