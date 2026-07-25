"""Independent controlled regime streams for Phase V validity tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ControlledRegimeStream:
    values: np.ndarray
    inputs: np.ndarray
    labels: np.ndarray
    boundaries: list[tuple[int, int, str]]
    metadata: dict[str, Any]


def _separation_scale(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return {"low": 0.10, "medium": 0.30, "high": 0.60}[str(value)]


def generate_controlled_regime_stream(
    length: int,
    *,
    regime_count: int = 2,
    regime_separation: str | float = "medium",
    return_probability: float = 0.5,
    dwell_length: int = 64,
    transition_type: str = "abrupt",
    observation_noise: float = 0.0,
    process_noise: float = 0.0,
    input_noise: float = 0.0,
    observability: str = "full",
    seed: int = 0,
) -> ControlledRegimeStream:
    if length < 16 or regime_count < 2:
        raise ValueError("length must be at least 16 and regime_count must be at least two")
    if not 0.0 <= return_probability <= 1.0:
        raise ValueError("return_probability must be in [0, 1]")
    if dwell_length < 2 or transition_type not in {"abrupt", "gradual"}:
        raise ValueError("invalid dwell length or transition type")
    if observability not in {"full", "partial", "hidden_driver"}:
        raise ValueError("unsupported observability")
    rng = np.random.default_rng(seed)
    labels = np.zeros(length, dtype=np.int64)
    current = 0
    visited = {current}
    for index in range(1, length):
        if index % dwell_length == 0:
            candidates = [regime for regime in range(regime_count) if regime != current]
            if rng.random() < return_probability:
                returning = sorted(visited.difference({current}))
                current = int(rng.choice(returning or candidates))
            else:
                current = int(rng.choice(candidates))
            visited.add(current)
        labels[index] = current
    separation = _separation_scale(regime_separation)
    raw_ar_coefficients = 0.65 + np.linspace(
        -separation, separation, regime_count
    )
    stability_margin = 0.05
    ar_coefficients = np.clip(
        raw_ar_coefficients,
        -1.0 + stability_margin,
        1.0 - stability_margin,
    )
    drivers = rng.normal(0.0, 1.0, size=length)
    if input_noise:
        drivers += rng.normal(0.0, input_noise, size=length)
    latent = np.zeros(length, dtype=np.float64)
    latent[0] = rng.normal(0.0, 0.1)
    for index in range(1, length):
        target_coeff = ar_coefficients[labels[index]]
        if transition_type == "gradual" and index > 0 and labels[index] != labels[index - 1]:
            target_coeff = 0.5 * target_coeff + 0.5 * ar_coefficients[labels[index - 1]]
        latent[index] = target_coeff * latent[index - 1] + 0.25 * np.tanh(drivers[index - 1]) + rng.normal(0.0, process_noise)
    observed = latent + rng.normal(0.0, observation_noise, size=length)
    observed_driver = drivers.copy()
    if observability == "partial":
        observed_driver = 0.5 * observed_driver
    elif observability == "hidden_driver":
        observed_driver.fill(0.0)
    boundaries = []
    start = 0
    for index in range(1, length):
        if labels[index] != labels[index - 1]:
            boundaries.append((start, index, str(labels[index - 1])))
            start = index
    boundaries.append((start, length, str(labels[-1])))
    return ControlledRegimeStream(
        values=observed.astype(np.float32),
        inputs=observed_driver.astype(np.float32),
        labels=labels,
        boundaries=boundaries,
        metadata={
            "seed": seed, "regime_count": regime_count, "regime_separation": regime_separation,
            "return_probability": return_probability, "dwell_length": dwell_length,
            "transition_type": transition_type, "observation_noise": observation_noise,
            "process_noise": process_noise, "input_noise": input_noise,
            "observability": observability,
            "raw_ar_coefficients": raw_ar_coefficients.tolist(),
            "ar_coefficients": ar_coefficients.tolist(),
            "stability_margin": stability_margin,
            "stability_clipped": bool(
                not np.array_equal(raw_ar_coefficients, ar_coefficients)
            ),
        },
    )


def make_independent_controlled_streams(
    *,
    lengths: dict[str, int] | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, ControlledRegimeStream]:
    lengths = lengths or {"train": 1200, "validation": 500, "test": 500, "prequential": 500}
    names = ("train", "validation", "test", "prequential")
    return {
        name: generate_controlled_regime_stream(int(lengths[name]), seed=seed + index * 1_000_003, **kwargs)
        for index, name in enumerate(names)
    }
from torch.utils.data import Dataset

class ControlledWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, stream: ControlledRegimeStream, window: int = 32) -> None:
        if len(stream.values) <= window:
            raise ValueError("controlled stream is too short for the requested window")
        self.values = stream.values
        self.inputs = stream.inputs
        self.labels = stream.labels
        self.window = window

    def __len__(self) -> int:
        return len(self.values) - self.window

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = np.stack([
            self.values[index:index + self.window],
            self.inputs[index:index + self.window],
        ], axis=-1)
        target = self.values[index + self.window]
        return torch.from_numpy(features), torch.tensor([target], dtype=torch.float32)
