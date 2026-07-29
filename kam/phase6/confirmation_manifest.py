"""Immutable, power-registered Phase 6 confirmation manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CAMPAIGN = "phase6_confirmation_v2"
TARGET_TOKENS = 50_000_000
CHECKPOINT_TOKENS = (1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000)
PRIMARY_SEEDS = tuple(range(71_001, 71_031))
REPLICATION_SEEDS = tuple(range(72_001, 72_025))
MECHANISM_SEEDS = tuple(range(73_001, 73_009))
SECONDARY_SEEDS = PRIMARY_SEEDS[:12]
EXPECTED_ROWS = 156


def _row_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "p6confirm_" + hashlib.sha256(payload).hexdigest()[:16]


def _supports(architecture: str) -> int:
    if architecture in {"T-KAM-F", "T-PKM"}:
        return 4096
    return 1024


def _optimization(architecture: str) -> str:
    return "alt_8_1" if architecture == "T-KAM-ALT" else "joint_sgd"


def _corpus(corpus_id: str) -> dict[str, Any]:
    if corpus_id == "tinystories_v2_128mib":
        root = Path("data/phase6_confirmation")
        return {
            "corpus_id": corpus_id,
            "corpus_train_path": str(root / "TinyStoriesV2-GPT4-train.128MiB.txt"),
            "corpus_validation_path": str(root / "TinyStories-valid.validation.txt"),
            "corpus_test_path": str(root / "TinyStories-valid.test.txt"),
        }
    if corpus_id == "tinyshakespeare":
        return {
            "corpus_id": corpus_id,
            "corpus_train_path": "data/tinyshakespeare.txt",
        }
    raise ValueError(f"unknown confirmation corpus: {corpus_id}")


def _language_row(
    *,
    cohort: str,
    corpus_id: str,
    architecture: str,
    seed: int,
    scientific_role: str,
    run_deletion_diagnostics: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "wave": "confirmation_v2",
        "lane": "confirmation_language",
        "task": "small_language",
        "cohort": cohort,
        "scientific_role": scientific_role,
        "architecture": architecture,
        "optimization": _optimization(architecture),
        "scale": "10M",
        "target_parameter_budget": 10_000_000,
        "seed": seed,
        "training_seed": seed,
        "data_seed": 900_000 + seed,
        "pair_id": f"{corpus_id}:{seed}",
        "sequence_length": 128,
        "batch_size": 16,
        "num_supports": _supports(architecture),
        "top_k": 4,
        "minimum_tokens": TARGET_TOKENS,
        "budget_mode": "matched_tokens",
        "validation_schedule": "registered_tokens",
        "validation_token_checkpoints": list(CHECKPOINT_TOKENS),
        "target_seconds": 3 * 60 * 60,
        "precision": "bf16",
        "cpus_per_row": 8,
        "ram_gb": 40,
        "run_deletion_diagnostics": run_deletion_diagnostics,
        "preregistered": True,
        **_corpus(corpus_id),
    }
    row["row_id"] = _row_id(row)
    return row


def build_confirmation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        for architecture in ("T-KAM-F", "T-WIDE"):
            rows.append(
                _language_row(
                    cohort="primary",
                    corpus_id="tinystories_v2_128mib",
                    architecture=architecture,
                    seed=seed,
                    scientific_role="primary_candidate" if architecture == "T-KAM-F" else "primary_comparator",
                    run_deletion_diagnostics=architecture == "T-KAM-F" and seed in PRIMARY_SEEDS[:6],
                )
            )
    for seed in SECONDARY_SEEDS:
        for architecture in ("T0", "T-PKM"):
            rows.append(
                _language_row(
                    cohort="secondary_control",
                    corpus_id="tinystories_v2_128mib",
                    architecture=architecture,
                    seed=seed,
                    scientific_role="secondary_control",
                )
            )
    for seed in REPLICATION_SEEDS:
        for architecture in ("T-KAM-F", "T-WIDE"):
            rows.append(
                _language_row(
                    cohort="replication",
                    corpus_id="tinyshakespeare",
                    architecture=architecture,
                    seed=seed,
                    scientific_role="replication_candidate" if architecture == "T-KAM-F" else "replication_comparator",
                )
            )
    for seed in MECHANISM_SEEDS:
        for architecture in ("T-KAM-F", "T-KAM-L", "T-KAM-ALT"):
            rows.append(
                _language_row(
                    cohort="mechanism",
                    corpus_id="tinystories_v2_128mib",
                    architecture=architecture,
                    seed=seed,
                    scientific_role="fixed_geometry_reference" if architecture == "T-KAM-F" else "learned_geometry_audit",
                    run_deletion_diagnostics=True,
                )
            )
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"confirmation row count changed: {len(rows)} != {EXPECTED_ROWS}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise AssertionError("confirmation manifest contains duplicate row IDs")
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows if rows is not None else build_confirmation_rows())
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized).encode("utf-8")
    destination.write_bytes(payload)
    return {
        "campaign": CAMPAIGN,
        "path": str(destination),
        "rows": len(materialized),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "primary_paired_seeds": len(PRIMARY_SEEDS),
        "replication_paired_seeds": len(REPLICATION_SEEDS),
        "mechanism_seeds": len(MECHANISM_SEEDS),
    }


__all__ = [
    "CAMPAIGN",
    "CHECKPOINT_TOKENS",
    "EXPECTED_ROWS",
    "MECHANISM_SEEDS",
    "PRIMARY_SEEDS",
    "REPLICATION_SEEDS",
    "SECONDARY_SEEDS",
    "TARGET_TOKENS",
    "build_confirmation_rows",
    "write_manifest",
]
