import numpy as np

from kam.data.controlled_regimes import (
    generate_controlled_regime_stream,
    make_independent_controlled_streams,
)


def test_controlled_stream_is_reproducible_and_has_explicit_boundaries():
    left = generate_controlled_regime_stream(200, seed=11, regime_count=4, dwell_length=25)
    right = generate_controlled_regime_stream(200, seed=11, regime_count=4, dwell_length=25)
    assert np.array_equal(left.values, right.values)
    assert np.array_equal(left.labels, right.labels)
    assert sum(stop - start for start, stop, _ in left.boundaries) == 200
    assert left.metadata["regime_count"] == 4


def test_independent_splits_use_distinct_stream_seeds():
    streams = make_independent_controlled_streams(seed=5, lengths={"train": 80, "validation": 80, "test": 80, "prequential": 80})
    assert streams["train"].metadata["seed"] != streams["validation"].metadata["seed"]
    assert not np.array_equal(streams["train"].values, streams["validation"].values)


def test_noise_and_observability_are_explicit_factors():
    clean = generate_controlled_regime_stream(200, seed=3, observation_noise=0.0)
    noisy = generate_controlled_regime_stream(200, seed=3, observation_noise=0.3)
    hidden = generate_controlled_regime_stream(200, seed=3, observability="hidden_driver")
    assert noisy.metadata["observation_noise"] == 0.3
    assert not np.array_equal(clean.values, noisy.values)
    assert np.all(hidden.inputs == 0.0)
