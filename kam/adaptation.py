from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


class LinearAdapter:
    """A scalar online linear readout with explicit predict-then-update semantics."""

    name = "base"

    def __init__(self, input_dim: int, output_dim: int = 1, device: torch.device | str = "cpu") -> None:
        self.linear = nn.Linear(input_dim, output_dim, device=device)

    @property
    def parameters(self):
        return self.linear.parameters()

    def predict(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def update(self, features: Tensor, targets: Tensor) -> None:
        raise NotImplementedError


class FrozenAdapter(LinearAdapter):
    name = "frozen"

    def update(self, features: Tensor, targets: Tensor) -> None:
        return None


class NLMSAdapter(LinearAdapter):
    name = "nlms"

    def __init__(self, input_dim: int, output_dim: int = 1, eta: float = 0.1, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        self.eta = eta
        self.eps = eps

    @torch.no_grad()
    def update(self, features: Tensor, targets: Tensor) -> None:
        if targets.ndim == 1:
            targets = targets[:, None]
        predictions = self.predict(features)
        errors = targets - predictions
        augmented = torch.cat(
            [features, torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)],
            dim=-1,
        )
        weights = torch.cat([self.linear.weight, self.linear.bias[:, None]], dim=1)
        denominator = self.eps + augmented.square().sum(dim=-1, keepdim=True)
        delta = self.eta * errors / denominator
        weights.add_(torch.einsum("bo,bf->of", delta, augmented) / features.shape[0])
        self.linear.weight.copy_(weights[:, :-1])
        self.linear.bias.copy_(weights[:, -1])


class SGDAdapter(LinearAdapter):
    name = "sgd"

    def __init__(self, input_dim: int, output_dim: int = 1, eta: float = 0.01, **kwargs) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=eta)

    def update(self, features: Tensor, targets: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(self.predict(features), targets)
        loss.backward()
        self.optimizer.step()


class RLSAdapter(LinearAdapter):
    name = "rls"

    def __init__(self, input_dim: int, output_dim: int = 1, forgetting: float = 1.0, delta: float = 1.0, **kwargs) -> None:
        super().__init__(input_dim, output_dim, **kwargs)
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("RLS forgetting must be in (0, 1].")
        self.forgetting = forgetting
        self.covariance = torch.eye(input_dim + 1, device=self.linear.weight.device) / delta

    @torch.no_grad()
    def update(self, features: Tensor, targets: Tensor) -> None:
        if targets.ndim == 1:
            targets = targets[:, None]
        if features.shape[0] != 1:
            for row, target in zip(features, targets):
                self.update(row[None, :], target[None, :])
            return
        augmented = torch.cat(
            [features, torch.ones(1, 1, device=features.device, dtype=features.dtype)], dim=-1
        ).T
        features = torch.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
        augmented = torch.nan_to_num(augmented, nan=0.0, posinf=1e3, neginf=-1e3)
        covariance = torch.nan_to_num(self.covariance, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        covariance = 0.5 * (covariance + covariance.T)
        gain_denominator = (self.forgetting + augmented.T @ covariance @ augmented).clamp_min(1e-4)
        gain = covariance @ augmented / gain_denominator
        weights = torch.cat([self.linear.weight, self.linear.bias[:, None]], dim=1).T
        prediction = weights.T @ augmented
        residual = torch.nan_to_num(targets.T - prediction, nan=0.0, posinf=1e3, neginf=-1e3)
        weights = weights + gain @ residual
        covariance = (covariance - gain @ augmented.T @ covariance) / self.forgetting
        self.covariance = torch.nan_to_num(0.5 * (covariance + covariance.T), nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        self.linear.weight.copy_(weights[:-1].T)
        self.linear.bias.copy_(weights[-1])


def make_adapter(name: str, input_dim: int, *, output_dim: int = 1, device: torch.device | str = "cpu", **kwargs) -> LinearAdapter:
    adapters = {"frozen": FrozenAdapter, "nlms": NLMSAdapter, "sgd": SGDAdapter, "rls": RLSAdapter}
    if name not in adapters:
        raise ValueError(f"Unsupported adapter: {name}")
    return adapters[name](input_dim, output_dim=output_dim, device=device, **kwargs)


@dataclass(frozen=True)
class PrequentialResult:
    predictions: Tensor
    targets: Tensor
    losses: Tensor
    metrics: dict[str, float]


def prequential_regression(
    feature_fn: Callable[[Tensor], Tensor],
    adapter: LinearAdapter,
    stream: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str = "cpu",
) -> PrequentialResult:
    """Score each sample before revealing its target and updating the adapter."""
    device = torch.device(device)
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    losses: list[Tensor] = []
    for inputs, target in stream:
        inputs = inputs.to(device)
        target = target.to(device)
        with torch.no_grad():
            features = feature_fn(inputs).detach()
            prediction = adapter.predict(features)
            loss = (prediction - target).square()
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
        losses.append(loss.detach().cpu())
        adapter.update(features, target)
    if not losses:
        raise ValueError("Prequential stream produced no samples.")
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    loss_tensor = torch.cat(losses)
    return PrequentialResult(
        predictions=prediction_tensor,
        targets=target_tensor,
        losses=loss_tensor,
        metrics={
            "mse": float(loss_tensor.mean()),
            "mae": float((prediction_tensor - target_tensor).abs().mean()),
            "samples": float(loss_tensor.numel()),
        },
    )
