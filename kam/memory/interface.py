from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class RouteResult:
    """Routing output in flattened token-major form.

    ``indices`` and ``weights`` are shaped ``[tokens, top_k]``.  Keeping the
    representation explicit makes exact/chunked/product-key routers directly
    comparable in Stage 0.
    """

    indices: Tensor
    weights: Tensor
    scores: Tensor
    num_supports: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    def validate(self, tokens: int | None = None, top_k: int | None = None) -> None:
        if self.indices.ndim != 2 or self.weights.shape != self.indices.shape:
            raise ValueError("route indices and weights must both have shape [tokens, top_k]")
        if self.scores.shape != self.indices.shape:
            raise ValueError("route scores must have shape [tokens, top_k]")
        if tokens is not None and self.indices.shape[0] != tokens:
            raise ValueError("route token count does not match query token count")
        if top_k is not None and self.indices.shape[1] != top_k:
            raise ValueError("route top_k does not match requested top_k")
        if self.indices.dtype not in (torch.int64, torch.long):
            raise ValueError("route indices must be int64")
        if self.indices.numel() and (self.indices.min() < 0 or self.indices.max() >= self.num_supports):
            raise ValueError("route indices contain an invalid support")


class MemoryLayer(nn.Module):
    """Protocol-like base class for memory layers.

    Geometry and algebra updates are deliberately separate so the optimizer
    ablations can exercise the same layer without reaching into parameters.
    """

    def route(self, query: Tensor, *, return_diagnostics: bool = False) -> RouteResult:
        raise NotImplementedError

    def retrieve(self, query: Tensor, route: RouteResult) -> Tensor:
        raise NotImplementedError

    def update_algebra(self, features: Tensor | None = None, targets: Tensor | None = None, **updates: Tensor) -> dict[str, Any]:
        del features, targets
        raise NotImplementedError

    def update_geometry(self, loss: Tensor | None = None, **kwargs: Any) -> dict[str, Any]:
        del loss
        raise NotImplementedError

    def diagnostics(self, route: RouteResult | None = None) -> dict[str, float]:
        return dict(route.diagnostics) if route is not None else {}
