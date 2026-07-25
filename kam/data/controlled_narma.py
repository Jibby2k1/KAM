"""Task-specific switching NARMA controlled stream."""
from __future__ import annotations
from typing import Any
import numpy as np
from .controlled_regimes import ControlledRegimeStream
from .stream_quality import assess_stream_quality, stream_quality_checks


DEFAULT_MAX_STABILITY_ATTEMPTS = 64
_RETRY_NAMESPACE = 0x4B414D


def _retry_seed(requested_seed: int, attempt: int) -> int:
    """Map a requested seed and retry index to a deterministic NumPy seed."""
    if attempt == 0:
        return int(requested_seed)
    return int(
        np.random.SeedSequence(
            [int(requested_seed), int(attempt), _RETRY_NAMESPACE]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _generate_controlled_narma_candidate(
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
    # Keep every regime inside a stable NARMA-like coefficient family.  The
    # previous 0.30 +/- 0.16 multiplier on y_t * sum(history) saturated the
    # safety clip and made NMSE primarily a denominator pathology.
    separation_map = {"low": 0.001, "medium": 0.003, "high": 0.005}
    separation = separation_map[str(regime_separation)] if str(regime_separation) in separation_map else float(regime_separation)
    driver = rng.uniform(0.0, 0.5, size=length)
    if input_noise:
        driver += rng.normal(0.0, input_noise, size=length)
    values = rng.uniform(0.0, 0.1, size=length).astype(np.float64)
    for index in range(order, length - 1):
        active = int(labels[index])
        gain = 0.05 + separation * (active - (regime_count - 1) / 2.0)
        history = np.clip(values[index - order:index], -2.0, 2.0)
        candidate = (
            0.30 * values[index]
            + gain * values[index] * float(history.sum())
            + 1.50 * driver[index - order + 1] * driver[index]
            + 0.10
        )
        candidate += rng.normal(0.0, process_noise)
        # Keep the controlled NARMA stream finite under the high-separation
        # factorial cells; this is a bounded state variable, not padding.
        values[index + 1] = np.clip(candidate, -5.0, 5.0)
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
         "observability": observability, "true_memory_horizon": order,
         "clip_boundary": 5.0, "narma_gain_separation": separation},
    )


def generate_controlled_narma_stream(
    length: int, *, seed: int = 0, regime_count: int = 3, order: int = 10,
    regime_separation: str | float = "medium", return_probability: float = 0.5,
    dwell_length: int = 64, transition_type: str = "abrupt",
    observation_noise: float = 0.0, process_noise: float = 0.0, input_noise: float = 0.0,
    observability: str = "full",
    max_stability_attempts: int = DEFAULT_MAX_STABILITY_ATTEMPTS,
    **kwargs: Any,
) -> ControlledRegimeStream:
    """Generate a stable stream while preserving valid requested-seed draws.

    Attempt zero is byte-for-byte the historical requested-seed generator.
    Only candidates that fail the registered stream-quality contract are
    retried.  The deterministic realized seed and attempt are recorded so a
    rejected draw never becomes an invisible change to the experiment.
    """
    if max_stability_attempts < 1:
        raise ValueError("max_stability_attempts must be at least one")
    last_quality: dict[str, Any] | None = None
    last_checks: dict[str, bool] | None = None
    for attempt in range(max_stability_attempts):
        realized_seed = _retry_seed(seed, attempt)
        stream = _generate_controlled_narma_candidate(
            length,
            seed=realized_seed,
            regime_count=regime_count,
            order=order,
            regime_separation=regime_separation,
            return_probability=return_probability,
            dwell_length=dwell_length,
            transition_type=transition_type,
            observation_noise=observation_noise,
            process_noise=process_noise,
            input_noise=input_noise,
            observability=observability,
            **kwargs,
        )
        quality = assess_stream_quality(
            stream.values,
            clip_boundary=stream.metadata["clip_boundary"],
        )
        checks = stream_quality_checks(quality)
        if all(checks.values()):
            stream.metadata.update(
                {
                    "seed": int(seed),
                    "requested_seed": int(seed),
                    "realized_seed": realized_seed,
                    "seed_attempt": attempt,
                    "max_stability_attempts": int(max_stability_attempts),
                    "stream_quality": quality,
                    "stream_quality_checks": checks,
                }
            )
            return stream
        last_quality = quality
        last_checks = checks
    failed = [name for name, passed in (last_checks or {}).items() if not passed]
    raise ValueError(
        "Controlled NARMA stream exhausted deterministic stability retries: "
        f"requested_seed={seed}, attempts={max_stability_attempts}, "
        f"failed_checks={failed}, last_quality={last_quality}"
    )
