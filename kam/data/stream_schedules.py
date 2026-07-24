from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .mackey_glass import generate_mackey_glass
from .narma import generate_narma


@dataclass(frozen=True)
class ScheduleSegment:
    name: str
    start: int
    stop: int
    fraction: float


def schedule_segments(length: int, schedule: Sequence[str]) -> list[ScheduleSegment]:
    if length < 1 or not schedule:
        raise ValueError("length and schedule must be positive.")
    boundaries = np.linspace(0, length, len(schedule) + 1, dtype=int)
    return [
        ScheduleSegment(name, int(boundaries[i]), int(boundaries[i + 1]), 0.5 if len(schedule) == 2 and i == 1 else 1.0)
        for i, name in enumerate(schedule)
    ]


def schedule_labels(length: int, schedule: Sequence[str]) -> np.ndarray:
    labels = np.empty(length, dtype=object)
    for segment in schedule_segments(length, schedule):
        labels[segment.start : segment.stop] = segment.name
    return labels


def generate_switching_mackey_glass(
    length: int,
    regimes: Mapping[str, Mapping[str, float]],
    schedule: Sequence[str],
    *,
    seed: int = 0,
    dt: float = 0.1,
    sample_every: int = 10,
    warmup: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous Mackey–Glass stream without resetting state."""
    if length < 2 or not schedule:
        raise ValueError("length and schedule must be valid.")
    max_tau = max(float(regimes[name]["tau"]) for name in regimes)
    delay_steps = int(round(max_tau / dt))
    total_samples = length + warmup
    total_steps = total_samples * sample_every
    rng = np.random.default_rng(seed)
    series = np.full(delay_steps + total_steps + 1, 1.2, dtype=np.float64)
    series[: delay_steps + 1] += rng.normal(0.0, 0.01, size=delay_steps + 1)
    labels = schedule_labels(total_samples, schedule)
    for step in range(delay_steps, delay_steps + total_steps):
        sample_index = min(total_samples - 1, step // sample_every)
        parameters = regimes[str(labels[sample_index])]
        delay = int(round(float(parameters["tau"]) / dt))
        delayed = series[step - delay]
        current = series[step]
        beta = float(parameters.get("beta", 0.2))
        gamma = float(parameters.get("gamma", 0.1))
        exponent = float(parameters.get("exponent", 10.0))
        derivative = beta * delayed / (1.0 + delayed**exponent) - gamma * current
        series[step + 1] = current + dt * derivative
    sampled = series[delay_steps + sample_every :: sample_every]
    sampled = sampled[warmup : warmup + length].astype(np.float32)
    return sampled, schedule_labels(length, schedule)


def generate_switching_narma(
    length: int,
    regimes: Mapping[str, Mapping[str, float]],
    schedule: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate recurring NARMA regimes while preserving the output history."""
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0.0, 0.5, size=length).astype(np.float64)
    values = np.zeros(length, dtype=np.float64)
    values[:20] = 0.1
    labels = schedule_labels(length, schedule)
    for index in range(20, length):
        parameters = regimes[str(labels[index])]
        order = int(parameters.get("order", 10.0))
        coefficient_scale = float(parameters.get("coefficient_scale", 1.0))
        history = values[index - order : index]
        values[index] = (
            0.3 * values[index - 1]
            + 0.05 * coefficient_scale * values[index - 1] * history.mean()
            + 1.5 * coefficient_scale * inputs[index - order] * inputs[index - 1]
            + 0.1
        )
        noise = float(parameters.get("noise_std", 0.0))
        if noise:
            values[index] += rng.normal(0.0, noise)
        values[index] = float(np.clip(values[index], -5.0, 5.0))
    return values.astype(np.float32), inputs.astype(np.float32), labels
