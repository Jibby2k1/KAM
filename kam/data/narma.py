from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_narma(
    length: int,
    *,
    order: int = 10,
    seed: int = 0,
    input_scale: float = 0.5,
    coefficient_scale: float = 1.0,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a bounded NARMA stream and its driving input."""
    if length < order + 2 or order < 2:
        raise ValueError("length must exceed order and order must be at least two.")
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0.0, input_scale, size=length).astype(np.float64)
    values = np.zeros(length, dtype=np.float64)
    values[:order] = 0.1
    for index in range(order, length):
        history = values[index - order : index]
        values[index] = (
            0.3 * values[index - 1]
            + 0.05 * coefficient_scale * values[index - 1] * history.sum()
            + 1.5 * coefficient_scale * inputs[index - order] * inputs[index - 1]
            + 0.1
        )
        if noise_std:
            values[index] += rng.normal(0.0, noise_std)
        values[index] = float(np.clip(values[index], -5.0, 5.0))
    return values.astype(np.float32), inputs.astype(np.float32)


class NARMADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windowed NARMA next-step regression."""

    def __init__(self, values: np.ndarray, inputs: np.ndarray, window: int = 32) -> None:
        if len(values) != len(inputs) or len(values) <= window:
            raise ValueError("NARMA values and inputs must have matching sufficient length.")
        self.values = np.asarray(values, dtype=np.float32)
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.window = window

    def __len__(self) -> int:
        return len(self.values) - self.window

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.values[index : index + self.window]
        driving = self.inputs[index : index + self.window]
        features = np.stack([values, driving], axis=-1)
        target = self.values[index + self.window]
        return torch.from_numpy(features), torch.tensor([target], dtype=torch.float32)


@dataclass(frozen=True)
class NARMASplits:
    train: NARMADataset
    validation: NARMADataset
    test: NARMADataset


def make_narma_splits(
    *, length: int = 12000, order: int = 10, window: int = 32, seed: int = 0
) -> NARMASplits:
    values, inputs = generate_narma(length, order=order, seed=seed)
    train_end = int(0.70 * length)
    validation_end = int(0.85 * length)
    train_values = values[:train_end]
    mean = float(train_values.mean())
    std = float(train_values.std() + 1e-8)
    values = (values - mean) / std
    return NARMASplits(
        train=NARMADataset(values[:train_end], inputs[:train_end], window),
        validation=NARMADataset(values[train_end - window : validation_end], inputs[train_end - window : validation_end], window),
        test=NARMADataset(values[validation_end - window :], inputs[validation_end - window :], window),
    )
