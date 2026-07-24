from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_effect(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("Paired samples must have matching shapes.")
    return left_array - right_array


def paired_permutation_pvalue(left: Sequence[float], right: Sequence[float], permutations: int = 10000, seed: int = 0) -> float:
    differences = paired_effect(left, right)
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, differences.size))
    null = np.abs((signs * differences[None, :]).mean(axis=1))
    return float((1 + (null >= observed).sum()) / (permutations + 1))


def paired_bootstrap_ci(left: Sequence[float], right: Sequence[float], *, confidence: float = 0.95, resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    differences = paired_effect(left, right)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
