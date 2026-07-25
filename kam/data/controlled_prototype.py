"""Distinct controlled prototype-regime generator for Phase V Stage 2."""
from __future__ import annotations
from typing import Any
from .controlled_regimes import ControlledRegimeStream, generate_controlled_regime_stream


def generate_prototype_stream(length: int, *, seed: int = 0, **kwargs: Any) -> ControlledRegimeStream:
    stream = generate_controlled_regime_stream(length, seed=seed, **kwargs)
    metadata = dict(stream.metadata)
    metadata.update({"task_generator": "controlled_prototype", "true_memory_horizon": int(kwargs.get("dwell_length", 64))})
    return ControlledRegimeStream(stream.values, stream.inputs, stream.labels, stream.boundaries, metadata)
