"""Numerical validity metrics and gates for controlled Stage 2 streams."""
from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_STREAM_LIMITS = {
    "minimum_target_variance": 1e-6,
    "maximum_absolute_state": 100.0,
    "maximum_saturation_fraction": 0.02,
    "minimum_unique_value_fraction": 0.01,
    "minimum_difference_variance": 1e-8,
    "maximum_absolute_lag1_autocorrelation": 0.9999,
}


def assess_stream_quality(
    values: np.ndarray,
    *,
    clip_boundary: float | None = None,
) -> dict[str, Any]:
    """Return explicit numerical-pathology metrics for one target stream."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    nonfinite_fraction = float(1.0 - finite.mean()) if flat.size else 1.0
    if finite_values.size:
        target_variance = float(np.var(finite_values))
        maximum_absolute_state = float(np.max(np.abs(finite_values)))
        rounded = np.round(finite_values, decimals=7)
        unique_value_fraction = float(np.unique(rounded).size / finite_values.size)
    else:
        target_variance = 0.0
        maximum_absolute_state = float("inf")
        unique_value_fraction = 0.0

    if finite_values.size > 1:
        differences = np.diff(finite_values)
        difference_variance = float(np.var(differences))
        left = finite_values[:-1] - finite_values[:-1].mean()
        right = finite_values[1:] - finite_values[1:].mean()
        denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
        lag1_autocorrelation = (
            float(np.dot(left, right) / denominator) if denominator > 0.0 else 1.0
        )
    else:
        difference_variance = 0.0
        lag1_autocorrelation = 1.0

    if clip_boundary is None or not finite_values.size:
        saturation_fraction = 0.0
    else:
        tolerance = max(1e-7, abs(float(clip_boundary)) * 1e-7)
        saturation_fraction = float(
            np.mean(np.abs(finite_values) >= abs(float(clip_boundary)) - tolerance)
        )

    return {
        "sample_count": int(flat.size),
        "nonfinite_fraction": nonfinite_fraction,
        "target_variance": target_variance,
        "maximum_absolute_state": maximum_absolute_state,
        "fraction_at_clip_boundary": saturation_fraction,
        "unique_value_fraction": unique_value_fraction,
        "difference_variance": difference_variance,
        "lag1_autocorrelation": lag1_autocorrelation,
    }


def stream_quality_checks(
    quality: dict[str, Any],
    *,
    limits: dict[str, float] | None = None,
) -> dict[str, bool]:
    """Evaluate the Stage 2 stream-stability contract."""
    thresholds = {**DEFAULT_STREAM_LIMITS, **(limits or {})}
    return {
        "finite_values": float(quality["nonfinite_fraction"]) == 0.0,
        "bounded_amplitude": float(quality["maximum_absolute_state"])
        <= thresholds["maximum_absolute_state"],
        "minimum_target_variance": float(quality["target_variance"])
        >= thresholds["minimum_target_variance"],
        "maximum_saturation_fraction": float(quality["fraction_at_clip_boundary"])
        <= thresholds["maximum_saturation_fraction"],
        "minimum_unique_value_fraction": float(quality["unique_value_fraction"])
        >= thresholds["minimum_unique_value_fraction"],
        "minimum_effective_variability": float(quality["difference_variance"])
        >= thresholds["minimum_difference_variance"],
        "nondegenerate_autocorrelation": abs(float(quality["lag1_autocorrelation"]))
        <= thresholds["maximum_absolute_lag1_autocorrelation"],
    }


def require_stable_stream(
    quality: dict[str, Any],
    *,
    stream_name: str,
    limits: dict[str, float] | None = None,
) -> None:
    checks = stream_quality_checks(quality, limits=limits)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"Controlled stream {stream_name!r} failed stability gates "
            f"{failed}: {quality}"
        )
