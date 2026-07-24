from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_mackey_glass(
    length: int,
    *,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: float = 10.0,
    tau: float = 17.0,
    dt: float = 0.1,
    sample_every: int = 10,
    warmup: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Generate an Euler-discretized Mackey-Glass trajectory.

    ``length`` and ``warmup`` are measured after downsampling.  The history is
    initialized near 1.2 with a small seeded perturbation.
    """
    if length < 2:
        raise ValueError("length must be at least two.")
    if dt <= 0 or sample_every < 1 or tau <= 0:
        raise ValueError("dt, sample_every, and tau must be positive.")

    rng = np.random.default_rng(seed)
    delay_steps = int(round(tau / dt))
    sampled_total = length + warmup
    integration_steps = sampled_total * sample_every
    series = np.full(delay_steps + integration_steps + 1, 1.2, dtype=np.float64)
    series[: delay_steps + 1] += rng.normal(0.0, 0.01, size=delay_steps + 1)

    for index in range(delay_steps, delay_steps + integration_steps):
        delayed = series[index - delay_steps]
        current = series[index]
        derivative = beta * delayed / (1.0 + delayed**exponent) - gamma * current
        series[index + 1] = current + dt * derivative

    sampled = series[delay_steps + sample_every :: sample_every]
    sampled = sampled[warmup : warmup + length]
    if sampled.shape[0] != length:
        raise RuntimeError("Mackey-Glass generator produced an unexpected length.")
    return sampled.astype(np.float32)


class MackeyGlassDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windowed next-step data with one token per lagged observation."""

    def __init__(self, standardized_series: np.ndarray, window: int = 32) -> None:
        if window < 2:
            raise ValueError("window must be at least two.")
        if len(standardized_series) <= window:
            raise ValueError("series is too short for the requested window.")
        self.series = np.asarray(standardized_series, dtype=np.float32)
        self.window = window
        self.lag = np.linspace(-1.0, 0.0, window, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.series) - self.window

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.series[index : index + self.window]
        tokens = np.stack([values, self.lag], axis=-1)
        target = self.series[index + self.window]
        return torch.from_numpy(tokens), torch.tensor([target], dtype=torch.float32)


@dataclass(frozen=True)
class MackeyGlassSplits:
    train: MackeyGlassDataset
    validation: MackeyGlassDataset
    test: MackeyGlassDataset
    mean: float
    std: float
    raw_train: np.ndarray
    raw_validation: np.ndarray
    raw_test: np.ndarray


def make_mackey_splits(
    *,
    total_length: int = 12000,
    window: int = 32,
    tau: float = 17.0,
    beta: float = 0.2,
    seed: int = 0,
) -> MackeyGlassSplits:
    if total_length < 1000:
        raise ValueError("total_length should be at least 1000 for meaningful splits.")
    raw = generate_mackey_glass(total_length, tau=tau, beta=beta, seed=seed)
    train_end = int(0.70 * total_length)
    validation_end = int(0.85 * total_length)
    raw_train = raw[:train_end]
    raw_validation = raw[train_end - window : validation_end]
    raw_test = raw[validation_end - window :]
    mean = float(raw_train.mean())
    std = float(raw_train.std() + 1e-8)

    def standardize(values: np.ndarray) -> np.ndarray:
        return (values - mean) / std

    return MackeyGlassSplits(
        train=MackeyGlassDataset(standardize(raw_train), window),
        validation=MackeyGlassDataset(standardize(raw_validation), window),
        test=MackeyGlassDataset(standardize(raw_test), window),
        mean=mean,
        std=std,
        raw_train=raw_train,
        raw_validation=raw_validation,
        raw_test=raw_test,
    )
