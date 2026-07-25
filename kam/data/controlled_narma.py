"""Task-specific switching NARMA controlled stream."""
from __future__ import annotations
from typing import Any
import numpy as np
from .controlled_regimes import ControlledRegimeStream


def generate_controlled_narma_stream(
    length: int, *, seed: int = 0, regime_count: int = 3, order: int = 10,
    regime_separation: str | float = "medium", return_probability: float = 0.5,
    dwell_length: int = 64, transition_type: str = "abrupt",
    observation_noise: float = 0.0, process_noise: float = 0.0, input_noise: float = 0.0,
    observability: str = "full", **_: Any,
) -> ControlledRegimeStream:
    if length < order + 8:
        raise ValueError("length is too short for the requested NARMA order")
    rng = np.random.default_rng(seed)
    labels = np.zeros(length, dtype=np.int64)
    current = 0
    visited = {current}
    for index in range(1, length):
        if index % dwell_length == 0:
            candidates = [value for value in range(regime_count) if value != current]
            if rng.random() < return_probability and visited - {current}:
                current = int(rng.choice(sorted(visited - {current})))
            else:
                current = int(rng.choice(candidates))
            visited.add(current)
        labels[index] = current
    separation = {"low": 0.02, "medium": 0.08, "high": 0.16}.get(str(regime_separation), float(regime_separation))
    driver = rng.uniform(0.0, 0.5, size=length)
    if input_noise:
        driver += rng.normal(0.0, input_noise, size=length)
    values = rng.uniform(0.0, 0.1, size=length).astype(np.float64)
    for index in range(order, length - 1):
        active = int(labels[index])
        gain = 0.3 + separation * (active - (regime_count - 1) / 2.0)
        history = values[index - order:index]
        values[index + 1] = (
            0.25 * values[index]
            + gain * values[index] * float(history.sum())
            + 0.08 * driver[index - order + 1]
            + 0.05
        )
        values[index + 1] += rng.normal(0.0, process_noise)
    observed = values + rng.normal(0.0, observation_noise, size=length)
    observed_driver = driver.copy()
    if observability in {"partial", "hidden_driver"}:
        if observability == "partial":
            observed_driver *= 0.5
        else:
            observed_driver.fill(0.0)
    boundaries = []
    start = 0
    for index in range(1, length):
        if labels[index] != labels[index - 1]:
            boundaries.append((start, index, str(labels[index - 1])))
            start = index
    boundaries.append((start, length, str(labels[-1])))
    return ControlledRegimeStream(
        observed.astype(np.float32), observed_driver.astype(np.float32), labels, boundaries,
        {"seed": seed, "task_generator": "switching_narma", "regime_count": regime_count,
         "order": order, "regime_separation": regime_separation,
         "return_probability": return_probability, "dwell_length": dwell_length,
         "transition_type": transition_type, "observation_noise": observation_noise,
         "process_noise": process_noise, "input_noise": input_noise,
         "observability": observability, "true_memory_horizon": order},
    )
