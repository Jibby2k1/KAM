"""Task-specific switching Mackey–Glass-like controlled stream."""
from __future__ import annotations
from typing import Any
import numpy as np
from .controlled_regimes import ControlledRegimeStream


def generate_controlled_mackey_glass_stream(
    length: int, *, seed: int = 0, regime_count: int = 3, regime_separation: str | float = "medium",
    return_probability: float = 0.5, dwell_length: int = 64, transition_type: str = "abrupt",
    observation_noise: float = 0.0, process_noise: float = 0.0, input_noise: float = 0.0,
    observability: str = "full", tau_values: list[int] | None = None, **_: Any,
) -> ControlledRegimeStream:
    if length < 32:
        raise ValueError("length must be at least 32")
    rng = np.random.default_rng(seed)
    labels = np.zeros(length, dtype=np.int64)
    current = 0
    visited = {current}
    transitions = []
    for index in range(1, length):
        if index % dwell_length == 0:
            candidates = [value for value in range(regime_count) if value != current]
            if rng.random() < return_probability and visited - {current}:
                current = int(rng.choice(sorted(visited - {current})))
            else:
                current = int(rng.choice(candidates))
            visited.add(current)
            transitions.append(index)
        labels[index] = current
    tau_values = tau_values or [8 + 4 * index for index in range(regime_count)]
    separation = {"low": 0.05, "medium": 0.12, "high": 0.24}.get(str(regime_separation), float(regime_separation))
    driver = rng.uniform(-1.0, 1.0, size=length)
    if input_noise:
        driver += rng.normal(0.0, input_noise, size=length)
    values = np.zeros(length, dtype=np.float64)
    values[: max(tau_values) + 1] = rng.normal(0.0, 0.05, size=max(tau_values) + 1)
    for index in range(max(tau_values) + 1, length):
        tau = int(tau_values[int(labels[index]) % len(tau_values)])
        coefficient = 0.65 + separation * (int(labels[index]) - (regime_count - 1) / 2.0)
        delayed = values[index - tau]
        previous = values[index - 1]
        if transition_type == "gradual" and index in transitions:
            coefficient = 0.5 * coefficient + 0.325
        values[index] = coefficient * previous + 0.18 * np.tanh(delayed) + 0.12 * np.tanh(driver[index - 1])
        values[index] += rng.normal(0.0, process_noise)
    observed = values + rng.normal(0.0, observation_noise, size=length)
    observed_driver = driver.copy()
    if observability == "partial":
        observed_driver *= 0.5
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
        observed.astype(np.float32), observed_driver.astype(np.float32), labels, boundaries,
        {"seed": seed, "task_generator": "switching_mackey_glass", "regime_count": regime_count,
         "regime_separation": regime_separation, "return_probability": return_probability,
         "dwell_length": dwell_length, "transition_type": transition_type,
         "observation_noise": observation_noise, "process_noise": process_noise,
         "input_noise": input_noise, "observability": observability,
         "true_memory_horizon": int(max(tau_values)), "tau_values": tau_values},
    )
