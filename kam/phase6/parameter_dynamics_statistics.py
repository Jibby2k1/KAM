"""Paired fixed-sample statistics for the Phase 6.1 mechanism study."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


def bootstrap_mean(values: list[float], seed: int = 6610) -> tuple[float, float]:
    if not values: return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    if len(array) == 1: return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed); draws = rng.choice(array, size=(20_000, len(array)), replace=True).mean(1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def paired_log_comparison(results: list[dict[str, Any]], candidate: str, comparator: str, *, metric: str,
                          checkpoint: int | None = None) -> dict[str, Any]:
    def value(row: dict[str, Any]) -> float | None:
        if checkpoint is None:
            candidate_value = row.get(metric); return float(candidate_value) if candidate_value is not None else None
        point = next((trace for trace in row.get("traces", []) if trace.get("checkpoint_target_tokens") == checkpoint), None)
        return float(point[metric]) if point is not None and point.get(metric) is not None else None
    candidate_values = {int(row["seed"]): value(row) for row in results if row.get("arm") == candidate}
    comparator_values = {int(row["seed"]): value(row) for row in results if row.get("arm") == comparator}
    seeds = sorted(seed for seed in candidate_values.keys() & comparator_values.keys()
                   if candidate_values[seed] is not None and comparator_values[seed] is not None)
    if not seeds: return {"candidate": candidate, "comparator": comparator, "paired_seeds": 0}
    ratios = np.asarray([math.log(float(candidate_values[seed]) / float(comparator_values[seed])) for seed in seeds])
    low, high = bootstrap_mean(ratios.tolist())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(seeds))), dtype=float)
    null = np.mean(signs * ratios, axis=1); observed = abs(float(ratios.mean()))
    return {"candidate": candidate, "comparator": comparator, "metric": metric, "checkpoint_tokens": checkpoint,
            "paired_seeds": len(seeds), "seed_ids": seeds, "geometric_relative_change": math.exp(float(ratios.mean())) - 1,
            "ci_low_relative_change": math.exp(low) - 1, "ci_high_relative_change": math.exp(high) - 1,
            "paired_sign_flip_p": float(np.mean(np.abs(null) >= observed)),
            "win_rate": float(np.mean([float(candidate_values[seed]) < float(comparator_values[seed]) for seed in seeds])),
            "standardized_effect_dz": float(ratios.mean() / ratios.std(ddof=1)) if len(ratios) > 1 and ratios.std(ddof=1) > 0 else None}


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get); adjusted: dict[str, float] = {}; running = 0.0
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * values[key])); adjusted[key] = running
    return adjusted


__all__ = ["bootstrap_mean", "holm_adjust", "paired_log_comparison"]
