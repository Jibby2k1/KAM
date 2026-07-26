from __future__ import annotations

import torch
from torch import Tensor, nn

from .interface import RouteResult


class VectorExperts(nn.Module):
    """One value vector per support."""

    def __init__(self, num_supports: int, d_model: int, scale: float = 0.02) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.randn(num_supports, d_model) * scale)

    def forward(self, query: Tensor, route: RouteResult) -> Tensor:
        del query
        values = self.values[route.indices]
        return (route.weights.unsqueeze(-1) * values).sum(dim=1)

    @property
    def active_parameters_per_token(self) -> int:
        return self.values.shape[-1]


class AffineExperts(nn.Module):
    """A small local affine map per support."""

    def __init__(self, num_supports: int, d_model: int, scale: float = 0.02) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_supports, d_model, d_model) * (scale / max(d_model, 1) ** 0.5))
        self.bias = nn.Parameter(torch.randn(num_supports, d_model) * scale)

    def forward(self, query: Tensor, route: RouteResult) -> Tensor:
        weights = self.weight[route.indices]
        bias = self.bias[route.indices]
        local = torch.einsum("nd,nkde->nke", query, weights) + bias
        return (route.weights.unsqueeze(-1) * local).sum(dim=1)

    @property
    def active_parameters_per_token(self) -> int:
        d = self.weight.shape[-1]
        return d * d + d


class LowRankExperts(nn.Module):
    """Low-rank local affine maps, useful for the budget-matched controls."""

    def __init__(self, num_supports: int, d_model: int, rank: int = 4, scale: float = 0.02) -> None:
        super().__init__()
        if rank <= 0 or rank > d_model:
            raise ValueError("rank must be in [1, d_model]")
        self.rank = rank
        self.left = nn.Parameter(torch.randn(num_supports, d_model, rank) * (scale / rank**0.5))
        self.right = nn.Parameter(torch.randn(num_supports, rank, d_model) * (scale / d_model**0.5))
        self.bias = nn.Parameter(torch.randn(num_supports, d_model) * scale)

    def forward(self, query: Tensor, route: RouteResult) -> Tensor:
        left = self.left[route.indices]
        right = self.right[route.indices]
        bias = self.bias[route.indices]
        latent = torch.einsum("nd,nkdr->nkr", query, left)
        local = torch.einsum("nkr,nkre->nke", latent, right) + bias
        return (route.weights.unsqueeze(-1) * local).sum(dim=1)

    @property
    def active_parameters_per_token(self) -> int:
        d, rank = self.left.shape[1], self.rank
        return 2 * d * rank + d


class SharedBasisExperts(nn.Module):
    """Affine experts represented as support coefficients over shared bases."""

    def __init__(self, num_supports: int, d_model: int, rank: int = 4, scale: float = 0.02) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("shared-basis rank must be positive")
        self.rank = rank
        self.coefficients = nn.Parameter(torch.randn(num_supports, rank) * scale)
        self.basis = nn.Parameter(torch.randn(rank, d_model, d_model) * (scale / d_model**0.5))
        self.bias = nn.Parameter(torch.randn(num_supports, d_model) * scale)

    def forward(self, query: Tensor, route: RouteResult) -> Tensor:
        coefficients = self.coefficients[route.indices]
        local_basis = torch.einsum("nkr, rde -> nkde", coefficients, self.basis)
        local = torch.einsum("nd,nkde->nke", query, local_basis) + self.bias[route.indices]
        return (route.weights.unsqueeze(-1) * local).sum(dim=1)

    @property
    def active_parameters_per_token(self) -> int:
        d = self.basis.shape[-1]
        return self.rank + d * d + d


class RoutesOnlyExperts(nn.Module):
    """Control that records routing while contributing a zero residual."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, query: Tensor, route: RouteResult) -> Tensor:
        del route
        return torch.zeros(query.shape[0], self.d_model, device=query.device, dtype=query.dtype)

    @property
    def active_parameters_per_token(self) -> int:
        return 0
