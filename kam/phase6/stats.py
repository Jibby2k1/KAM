from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def paired_effect(baseline: Iterable[float], candidate: Iterable[float]) -> dict[str, float]:
    base = np.asarray(list(baseline), dtype=float)
    new = np.asarray(list(candidate), dtype=float)
    if base.shape != new.shape or base.size == 0:
        raise ValueError("paired arrays must have equal non-zero length")
    delta = new - base
    return {
        "n": float(delta.size),
        "mean_difference": float(delta.mean()),
        "relative_improvement": float(delta.mean() / (np.abs(base).mean() + 1e-12)),
        "median_difference": float(np.median(delta)),
        "std_difference": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
        "effect_size_dz": float(delta.mean() / (delta.std(ddof=1) + 1e-12)) if delta.size > 1 else 0.0,
    }


def bootstrap_ci(values: Iterable[float], *, statistic: str = "mean", replicates: int = 2000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, values.size), replace=True)
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError("statistic must be mean or median")
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def exact_permutation_test(baseline: Iterable[float], candidate: Iterable[float], *, seed: int = 0, permutations: int = 10000) -> dict[str, float]:
    base = np.asarray(list(baseline), dtype=float)
    new = np.asarray(list(candidate), dtype=float)
    if base.shape != new.shape or base.size == 0:
        raise ValueError("paired arrays must have equal non-zero length")
    observed = float(np.mean(new - base))
    if base.size <= 20:
        import itertools

        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=base.size)), dtype=float)
        permutations = len(signs)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, base.size))
    null = np.mean(signs * (new - base), axis=1)
    p_value = float((np.abs(null) >= abs(observed)).mean())
    return {"observed_difference": observed, "p_value": p_value, "permutations": float(permutations)}


def equivalence_test(baseline: Iterable[float], candidate: Iterable[float], margin: float, *, alpha: float = 0.05) -> dict[str, float | bool]:
    base = np.asarray(list(baseline), dtype=float)
    new = np.asarray(list(candidate), dtype=float)
    delta = new - base
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / math.sqrt(delta.size)) if delta.size > 1 else 0.0
    critical = 1.96 if alpha == 0.05 else 1.96
    lower, upper = mean - critical * standard_error, mean + critical * standard_error
    return {"equivalent": bool(lower > -margin and upper < margin), "margin": float(margin), "ci_lower": lower, "ci_upper": upper}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        corrected = min(1.0, (len(ordered) - index) * p_value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def aggregate_by_training_seed(rows: list[dict], *, metric: str, seed_key: str = "seed", stream_key: str = "stream") -> list[dict]:
    grouped: dict[tuple[object, object], list[float]] = defaultdict(list)
    for row in rows:
        if metric in row:
            grouped[(row.get(seed_key), row.get(stream_key))].append(float(row[metric]))
    return [{seed_key: seed, stream_key: stream, metric: float(np.mean(values)), "n_streams": len(values)} for (seed, stream), values in grouped.items()]


__all__ = ["aggregate_by_training_seed", "bootstrap_ci", "equivalence_test", "exact_permutation_test", "holm_adjust", "paired_effect"]
