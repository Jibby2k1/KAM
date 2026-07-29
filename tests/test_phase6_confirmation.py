from __future__ import annotations

from copy import deepcopy
import json

from kam.phase6.confirmation_analysis import evaluate_confirmation, final_aggregate, paired_log_ratio
from kam.phase6.confirmation_manifest import (
    EXPECTED_ROWS,
    MECHANISM_SEEDS,
    PRIMARY_SEEDS,
    REPLICATION_SEEDS,
    build_confirmation_rows,
)
from kam.phase6.overnight_runner import _language_corpus
from kam.phase6.confirmation_repair import exact_subset


def test_confirmation_manifest_is_fixed_paired_and_fresh() -> None:
    rows = build_confirmation_rows()
    assert len(rows) == EXPECTED_ROWS == 156
    assert len({row["row_id"] for row in rows}) == EXPECTED_ROWS
    assert not ({3301, 3302, 3303} & {int(row["seed"]) for row in rows})
    for corpus_id, seeds in (("tinystories_v2_128mib", PRIMARY_SEEDS), ("tinyshakespeare", REPLICATION_SEEDS)):
        for seed in seeds:
            pair = [row for row in rows if row["corpus_id"] == corpus_id and row["seed"] == seed and row["architecture"] in {"T-KAM-F", "T-WIDE"}]
            assert len(pair) == 2
            assert len({row["data_seed"] for row in pair}) == 1
            assert len({row["minimum_tokens"] for row in pair}) == 1


def test_registered_corpus_uses_disjoint_files(tmp_path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    test = tmp_path / "test.txt"
    train.write_bytes(b"a" * 5000)
    validation.write_bytes(b"b" * 5000)
    test.write_bytes(b"c" * 5000)
    tokens, metadata = _language_corpus(
        {
            "corpus_id": "fixture",
            "corpus_train_path": str(train),
            "corpus_validation_path": str(validation),
            "corpus_test_path": str(test),
        }
    )
    assert tokens.numel() == 15_000
    assert metadata["train_range"] == [0, 5000]
    assert metadata["validation_range"] == [5000, 10_000]
    assert metadata["test_range"] == [10_000, 15_000]
    assert metadata["train_sha256"] != metadata["validation_sha256"] != metadata["test_sha256"]
    assert metadata["split_overlap"] is False


def _synthetic_completed_rows() -> tuple[list[dict], list[dict]]:
    manifest = build_confirmation_rows()
    completed = []
    for row in manifest:
        architecture = row["architecture"]
        if architecture == "T-KAM-F":
            loss = 0.90
        elif architecture == "T-WIDE":
            loss = 1.00
        elif architecture == "T0":
            loss = 1.10
        elif architecture == "T-PKM":
            loss = 1.05
        else:
            loss = 0.95
        learned = architecture in {"T-KAM-L", "T-KAM-ALT"}
        subrun = {
            "training_seed": row["seed"],
            "test_loss": loss + (row["seed"] % 7) * 0.0001,
            "validation_loss": loss,
            "best_validation_loss": loss - 0.01,
            "generalization_gap": 0.01,
            "tokens": 50_001_000,
            "target_tokens_resolved": 50_000_000,
            "wall_seconds": 100.0,
            "tokens_per_second": 500_000.0,
            "estimated_training_flops": 1e15,
            "active_parameters_per_token": 4_000_000,
            "total_parameters": 10_000_000,
            "peak_vram_bytes": 1_000_000,
            "dataset_sha256": f"hash-{row['corpus_id']}",
            "geometry_steps": 100 if learned else 0,
            "geometry_freeze_tokens": 40_000_000 if learned else None,
            "geometry_frozen_for_final_tuning": True,
            "post_freeze_geometry_drift": 0.0,
            "loss_history": (
                [
                    {"tokens": 20_000_000, "memory_key_grad_norm": 1.0, "geometry_frozen": 0.0},
                    {"tokens": 40_000_000, "memory_key_grad_norm": 0.0, "geometry_frozen": 1.0},
                    {"tokens": 50_000_000, "memory_key_grad_norm": 0.0, "geometry_frozen": 1.0},
                ]
                if learned
                else []
            ),
            "deletion_metrics": [],
        }
        completed.append({**row, "status": "pass", "failure_category": None, "metrics": {"subruns": [subrun]}})
    return completed, manifest


def test_confirmation_locked_decision_requires_primary_replication_and_lifecycle() -> None:
    rows, manifest = _synthetic_completed_rows()
    result = evaluate_confirmation(rows, manifest)
    assert result["decision"] == "PROMOTE_FIXED_KEY_FAST_ALGEBRA"
    assert result["primary_pass"]
    assert result["replication_pass"]
    assert result["learned_memory_lifecycle_pass"]
    assert result["guardrails"]["passed"]


def test_confirmation_blocks_data_order_mismatch() -> None:
    rows, manifest = _synthetic_completed_rows()
    corrupted = deepcopy(manifest)
    target = next(row for row in corrupted if row["corpus_id"] == "tinystories_v2_128mib" and row["seed"] == PRIMARY_SEEDS[0] and row["architecture"] == "T-WIDE")
    target["data_seed"] += 1
    result = evaluate_confirmation(rows, corrupted)
    assert result["decision"] == "BLOCKED_INVALID_CONFIRMATION"
    assert not result["guardrails"]["checks"]["paired_data_order"]


def test_primary_uses_exact_seed_identity_and_practical_margin() -> None:
    observations = []
    for seed in PRIMARY_SEEDS:
        observations.append({"corpus_id": "x", "architecture": "T-KAM-F", "training_seed": seed, "test_loss": 0.99})
        observations.append({"corpus_id": "x", "architecture": "T-WIDE", "training_seed": seed, "test_loss": 1.00})
    result = paired_log_ratio(
        list(reversed(observations)),
        corpus_id="x",
        candidate="T-KAM-F",
        comparator="T-WIDE",
        seeds=PRIMARY_SEEDS,
        comparison="fixture",
    )
    assert result["seed_ids"] == list(PRIMARY_SEEDS)
    assert result["paired_seeds"] == 30
    assert result["ci_high_relative_change"] > -0.02


def test_mechanism_seed_count_is_prespecified() -> None:
    rows = build_confirmation_rows()
    learned = [row for row in rows if row["cohort"] == "mechanism" and row["architecture"] in {"T-KAM-L", "T-KAM-ALT"}]
    assert len(learned) == 2 * len(MECHANISM_SEEDS) == 16


def test_timeout_repair_is_an_exact_manifest_subset(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = build_confirmation_rows()
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, audit = exact_subset(manifest, [133, 134, 136, 137])
    assert selected == [rows[index] for index in [133, 134, 136, 137]]
    assert audit["scientific_fields_modified"] is False
    assert audit["source_indices"] == [133, 134, 136, 137]


def test_confirmation_report_builds_all_artifacts(tmp_path) -> None:
    rows, manifest = _synthetic_completed_rows()
    run_root = tmp_path / "results"
    report_root = tmp_path / "reports"
    row_root = run_root / "rows" / "confirmation_v2"
    row_root.mkdir(parents=True)
    (run_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    for row in rows:
        (row_root / f"{row['row_id']}.json").write_text(
            json.dumps(row, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = final_aggregate(run_root, report_root)
    assert summary["decision"] == "PROMOTE_FIXED_KEY_FAST_ALGEBRA"
    assert (run_root / "final_summary.json").is_file()
    assert (run_root / "confirmation_seed_metrics.parquet").is_file()
    assert (run_root / "confirmation_comparisons.parquet").is_file()
    assert (run_root / "mechanism_audit.parquet").is_file()
    assert (report_root / "CONFIRMATION_REPORT.md").is_file()
    assert len(summary["figures"]) == 6
    assert all(__import__("pathlib").Path(path).is_file() for path in summary["figures"])
