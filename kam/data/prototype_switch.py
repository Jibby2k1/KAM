from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .switching import SwitchingRegressionDataset, SwitchingRegressionSplits
from .stream_schedules import schedule_labels


def generate_prototype_switch(
    length: int,
    *,
    schedule: Sequence[str] = ("A", "B", "A"),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small recurring nonlinear prototype-switch stream.

    The two regimes use different nonlinear autoregressive maps and share the
    same observed scalar/input interface. This is intentionally a diagnostic
    task, not a physical-system claim.
    """
    if length < 32 or not schedule:
        raise ValueError("length and schedule must be valid")
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-1.0, 1.0, size=length).astype(np.float64)
    values = np.zeros(length, dtype=np.float64)
    values[:2] = rng.normal(0.0, 0.02, size=2)
    labels = schedule_labels(length, schedule)
    for index in range(2, length):
        regime = str(labels[index])
        previous = values[index - 1]
        if regime == "A":
            value = 0.72 * previous + 0.16 * np.sin(2.0 * inputs[index - 1]) + 0.06 * previous**2
        else:
            value = 0.43 * previous + 0.24 * np.tanh(2.5 * inputs[index - 1]) + 0.10 * previous * values[index - 2]
        values[index] = float(np.clip(value, -3.0, 3.0))
    return values.astype(np.float32), inputs.astype(np.float32), labels


def make_prototype_switch_splits(
    *, length: int = 6000, window: int = 32, schedule: Sequence[str] = ("A", "B", "A"), seed: int = 0
) -> SwitchingRegressionSplits:
    values, inputs, labels = generate_prototype_switch(length, schedule=schedule, seed=seed)
    train_end = int(0.70 * length)
    validation_end = int(0.85 * length)
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std() + 1e-8)

    def part(start: int, stop: int) -> SwitchingRegressionDataset:
        start = max(0, start - window)
        return SwitchingRegressionDataset(
            values[start:stop], inputs[start:stop], labels[start:stop],
            window=window, value_mean=mean, value_std=std,
        )

    return SwitchingRegressionSplits(
        train=part(0, train_end),
        validation=part(train_end, validation_end),
        test=part(validation_end, length),
        labels=labels, raw_values=values, raw_inputs=inputs,
        value_mean=mean, value_std=std,
    )
