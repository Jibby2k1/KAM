from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .stream_schedules import generate_switching_mackey_glass, generate_switching_narma


class SwitchingRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windowed next-step regression over one continuous, labeled stream."""

    def __init__(
        self,
        values: np.ndarray,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        window: int,
        value_mean: float = 0.0,
        value_std: float = 1.0,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels)
        if values.ndim != 1 or inputs.ndim != 1 or len(values) != len(inputs) or len(values) != len(labels):
            raise ValueError("switching streams must have equal one-dimensional arrays")
        if len(values) <= window:
            raise ValueError("switching stream is too short for the requested window")
        self.values = (values - float(value_mean)) / max(float(value_std), 1e-8)
        self.inputs = inputs
        self.labels = labels
        self.window = window
        self.value_mean = float(value_mean)
        self.value_std = float(value_std)

    def __len__(self) -> int:
        return len(self.values) - self.window

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.values[index : index + self.window]
        driving = self.inputs[index : index + self.window]
        features = np.stack([values, driving], axis=-1)
        target = self.values[index + self.window]
        return torch.from_numpy(features), torch.tensor([target], dtype=torch.float32)

    def target_regimes(self) -> np.ndarray:
        return self.labels[self.window :]


@dataclass(frozen=True)
class SwitchingRegressionSplits:
    train: SwitchingRegressionDataset
    validation: SwitchingRegressionDataset
    test: SwitchingRegressionDataset
    labels: np.ndarray
    raw_values: np.ndarray
    raw_inputs: np.ndarray
    value_mean: float
    value_std: float


def _split_stream(
    values: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    window: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SwitchingRegressionSplits:
    length = len(values)
    train_end = int(length * train_fraction)
    validation_end = int(length * (train_fraction + validation_fraction))
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std() + 1e-8)

    def part(start: int, stop: int) -> SwitchingRegressionDataset:
        start = max(0, start - window)
        return SwitchingRegressionDataset(
            values[start:stop],
            inputs[start:stop],
            labels[start:stop],
            window=window,
            value_mean=mean,
            value_std=std,
        )

    return SwitchingRegressionSplits(
        train=part(0, train_end),
        validation=part(train_end, validation_end),
        test=part(validation_end, length),
        labels=labels,
        raw_values=values,
        raw_inputs=inputs,
        value_mean=mean,
        value_std=std,
    )


def make_switching_mackey_splits(
    *,
    total_length: int = 6000,
    window: int = 32,
    regimes: Mapping[str, Mapping[str, float]] | None = None,
    schedule: Sequence[str] = ("A", "B", "A"),
    seed: int = 0,
) -> SwitchingRegressionSplits:
    regimes = regimes or {
        "A": {"tau": 17.0, "beta": 0.20},
        "B": {"tau": 20.0, "beta": 0.20},
        "C": {"tau": 17.0, "beta": 0.22},
    }
    values, labels = generate_switching_mackey_glass(
        total_length,
        regimes=regimes,
        schedule=schedule,
        seed=seed,
    )
    inputs = np.zeros_like(values)
    inputs[1:] = values[:-1]
    return _split_stream(values, inputs, labels, window=window)


def make_switching_narma_splits(
    *,
    total_length: int = 6000,
    window: int = 32,
    regimes: Mapping[str, Mapping[str, float]] | None = None,
    schedule: Sequence[str] = ("A", "B", "A"),
    seed: int = 0,
) -> SwitchingRegressionSplits:
    regimes = regimes or {
        "A": {"order": 10, "coefficient_scale": 1.0, "noise_std": 0.0},
        "B": {"order": 10, "coefficient_scale": 1.05, "noise_std": 0.0},
        "C": {"order": 20, "coefficient_scale": 1.0, "noise_std": 0.0},
    }
    values, inputs, labels = generate_switching_narma(
        total_length,
        regimes=regimes,
        schedule=schedule,
        seed=seed,
    )
    return _split_stream(values, inputs, labels, window=window)
