from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.readout = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden, _ = self.gru(self.embedding(inputs))
        return self.readout(hidden)


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, d_model, num_layers=num_layers, batch_first=True)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden, _ = self.gru(inputs)
        return self.readout(hidden[:, -1, :])


class MLPRegressor(nn.Module):
    def __init__(self, window: int, input_dim: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


@dataclass
class BudgetedKLMSRegressor:
    """A small quantized, fixed-budget KLMS baseline for streaming regression."""

    budget: int = 128
    sigma: float = 2.0
    eta: float = 0.2
    novelty: float = 0.5

    def __post_init__(self) -> None:
        self.centers: list[np.ndarray] = []
        self.coefficients: list[float] = []
        self.usage: list[int] = []

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        if not self.centers:
            return np.zeros(0, dtype=np.float64)
        centers = np.stack(self.centers)
        distance_sq = np.square(centers - x[None, :]).sum(axis=1)
        return np.exp(-0.5 * distance_sq / (self.sigma**2))

    def predict(self, x: np.ndarray) -> float:
        kernels = self._kernel(np.asarray(x, dtype=np.float64))
        if kernels.size == 0:
            return 0.0
        return float(kernels @ np.asarray(self.coefficients))

    def update(self, x: np.ndarray, y: float) -> float:
        x = np.asarray(x, dtype=np.float64)
        prediction = self.predict(x)
        error = float(y - prediction)
        if not self.centers:
            self.centers.append(x.copy())
            self.coefficients.append(self.eta * error)
            self.usage.append(1)
            return error

        centers = np.stack(self.centers)
        distances = np.linalg.norm(centers - x[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self.novelty:
            self.coefficients[nearest] += self.eta * error
            self.usage[nearest] += 1
        elif len(self.centers) < self.budget:
            self.centers.append(x.copy())
            self.coefficients.append(self.eta * error)
            self.usage.append(1)
        else:
            replace = int(np.argmin(self.usage))
            self.centers[replace] = x.copy()
            self.coefficients[replace] = self.eta * error
            self.usage[replace] = 1
        return error
