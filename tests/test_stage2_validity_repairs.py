from pathlib import Path

import numpy as np

from kam.capacity import active_parameter_count
from kam.data.controlled_narma import (
    _generate_controlled_narma_candidate,
    generate_controlled_narma_stream,
)
from kam.data.controlled_regimes import generate_controlled_regime_stream
from kam.data.controlled_symbolic_regime import ControlledSymbolicRegimeDataset
from kam.data.stream_quality import assess_stream_quality, stream_quality_checks
from kam.factory import make_model
from kam.phase4.table import read_table
from kam.phase5.stage2_manifest import build_symbolic_rows, factorial_designs


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


def test_controlled_narma_retries_bad_seed_deterministically() -> None:
    requested_seed = 41271
    rejected = _generate_controlled_narma_candidate(800, seed=requested_seed)
    rejected_quality = assess_stream_quality(
        rejected.values, clip_boundary=rejected.metadata["clip_boundary"]
    )
    assert not all(stream_quality_checks(rejected_quality).values())

    first = generate_controlled_narma_stream(800, seed=requested_seed)
    second = generate_controlled_narma_stream(800, seed=requested_seed)
    assert first.metadata["requested_seed"] == requested_seed
    assert first.metadata["realized_seed"] != requested_seed
    assert first.metadata["seed_attempt"] > 0
    assert first.metadata["realized_seed"] == second.metadata["realized_seed"]
    np.testing.assert_array_equal(first.values, second.values)
    assert all(first.metadata["stream_quality_checks"].values())


def test_controlled_narma_preserves_valid_requested_seed() -> None:
    stream = generate_controlled_narma_stream(800, seed=7)
    assert stream.metadata["requested_seed"] == 7
    assert stream.metadata["realized_seed"] == 7
    assert stream.metadata["seed_attempt"] == 0


def test_all_stage2_narma_campaign_and_heldout_requests_are_stable() -> None:
    manifest_root = Path("results/phase5/stage2/manifests")
    for stage in ("stage2B_capacity", "stage2C_factorial"):
        rows = read_table(manifest_root / f"{stage}.jsonl")
        representatives = {
            (row["cell"], row["seed_index"]): row
            for row in rows
            if row["task"] == "switching_narma_controlled"
        }
        for row in representatives.values():
            kwargs = {
                "regime_count": row["regime_count"],
                "order": row["order"],
                "regime_separation": row["regime_separation"],
                "return_probability": row["return_probability"],
                "dwell_length": row["dwell_length"],
                "transition_type": row["transition_type"],
                "observation_noise": row["observation_noise"],
                "process_noise": row["process_noise"],
                "input_noise": row["input_noise"],
                "observability": row["observability"],
            }
            split_lengths = (
                row["train_length"],
                row["validation_length"],
                row["test_length"],
                row["prequential_length"],
            )
            requested_roots = [row["seed"]]
            requested_roots.extend(
                row["seed"] + 7_000_003 * (index + 1)
                for index in range(row["heldout_streams"])
            )
            for requested_root in requested_roots:
                for split_index, length in enumerate(split_lengths):
                    requested_seed = requested_root + split_index * 1_000_003
                    stream = generate_controlled_narma_stream(
                        length, seed=requested_seed, **kwargs
                    )
                    assert all(stream.metadata["stream_quality_checks"].values())


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


def test_every_symbolic_manifest_row_matches_runtime_capacity() -> None:
    rows = build_symbolic_rows()
    assert {row["vocab_size"] for row in rows} == {13, 16}
    for row in rows:
        spec = {
            "model_name": row["variant"],
            "task_type": "language",
            "d_model": row["d_model"],
            "num_heads": row["num_heads"],
            "num_layers": row["num_layers"],
            "num_supports": row["num_supports"],
            "max_seq_len": row["seq_len"] + 1,
            "vocab_size": row["vocab_size"],
            "route_features": row["route_features"],
            "route_projection_dim": row["route_projection_dim"],
            "memory_output": row["memory_output"],
            "return_attention_for_diagnostics": row[
                "return_attention_for_diagnostics"
            ],
            "append_routes_to_readout": row["append_routes_to_readout"],
            "apply_memory_residual": row["apply_memory_residual"],
            "ffn_expansion": row["ffn_expansion"],
        }
        model = make_model(spec)
        assert active_parameter_count(model) == row["resolved_active_parameters"]
