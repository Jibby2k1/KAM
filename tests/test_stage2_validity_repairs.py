import numpy as np

from kam.data.controlled_narma import generate_controlled_narma_stream
from kam.data.controlled_regimes import generate_controlled_regime_stream
from kam.data.controlled_symbolic_regime import ControlledSymbolicRegimeDataset
from kam.data.stream_quality import assess_stream_quality, stream_quality_checks
from kam.phase5.stage2_manifest import factorial_designs


def test_high_separation_prototype_is_stability_constrained() -> None:
    stream = generate_controlled_regime_stream(
        2000, regime_count=3, regime_separation="high", seed=7
    )
    assert max(abs(value) for value in stream.metadata["ar_coefficients"]) <= 0.95
    assert stream.metadata["stability_clipped"]
    assert all(stream_quality_checks(assess_stream_quality(stream.values)).values())


def test_controlled_narma_is_not_clip_saturated() -> None:
    stream = generate_controlled_narma_stream(
        2000, regime_count=3, regime_separation="high", seed=7
    )
    quality = assess_stream_quality(
        stream.values, clip_boundary=stream.metadata["clip_boundary"]
    )
    assert quality["fraction_at_clip_boundary"] <= 0.02
    assert all(stream_quality_checks(quality).values())


def test_symbolic_return_probability_causes_actual_recurrence() -> None:
    dataset = ControlledSymbolicRegimeDataset(
        1,
        sequence_length=24,
        regime_count=3,
        transition_entropy=1.0,
        return_probability=1.0,
        explicit_regime_token=True,
        seed=9,
    )
    sample = dataset[0]
    regimes = sample["metadata"].numpy()
    assert np.all(regimes[1:] != regimes[:-1])
    assert len(np.unique(regimes)) >= 2
    assert np.all(sample["inputs"].numpy() >= dataset.regime_offset)


def test_factorial_is_balanced_and_fidelity_is_not_confounded() -> None:
    designs = factorial_designs()
    assert len(designs) == 18
    for field in (
        "return_probability",
        "regime_separation",
        "observability",
        "observation_noise",
        "center_initialization",
        "num_supports",
    ):
        counts = {}
        for row in designs:
            counts[row[field]] = counts.get(row[field], 0) + 1
        assert set(counts.values()) == {6}
    process_counts = {}
    for row in designs:
        process_counts[row["process_noise"]] = (
            process_counts.get(row["process_noise"], 0) + 1
        )
    assert set(process_counts.values()) == {9}
