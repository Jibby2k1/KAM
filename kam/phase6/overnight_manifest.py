"""Deterministic manifests for the Phase 6 four-L4 overnight campaign.

The campaign uses fixed row counts so the complete Slurm dependency graph can
be submitted before calibration finishes.  Runtime budgets are resolved from
the preflight calibration file by :mod:`kam.phase6.overnight_runner`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CAMPAIGN = "phase6_overnight_4xl4_quality_campaign"
PREFLIGHT_ROWS = 4
WAVE1_ROWS = 32
WAVE2_ROWS = 16
WAVE3_ROWS = 8
TARGET_SECONDS = {"preflight": 20 * 60, "wave1": 25 * 60, "wave2": 64 * 60, "wave3": 105 * 60}
WAVE1_TIMEOUT_REPAIR_INDICES = frozenset({4, 5, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24})
RETRIEVAL_MINIMUM_TOKENS = 5_000_000


def _row_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "p6night_" + hashlib.sha256(payload).hexdigest()[:16]


def _finalize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = {
            "campaign": CAMPAIGN,
            "design_index": index,
            "precision": "bf16",
            "cpus_per_row": 8,
            "ram_gb": 40,
            **source,
        }
        row["row_id"] = _row_id(row)
        result.append(row)
    if len({row["row_id"] for row in result}) != len(result):
        raise ValueError("overnight manifest contains duplicate row IDs")
    return result


def _language_supports(architecture: str) -> int:
    if architecture == "T-MEMTOK":
        return 32
    if architecture in {"T-KAM-L", "T-KAM-ALT", "T-KAM-VP"}:
        return 1024
    if architecture in {"T-PKM", "T-KAM-F"}:
        return 4096
    return 1024


def _retrieval_supports(architecture: str, sequence_length: int) -> int:
    if architecture == "T-MEMTOK":
        return 32
    if architecture == "T-KAM-L" or (architecture == "T-KAM-F" and sequence_length >= 512):
        return 1024
    if architecture in {"T-PKM", "T-KAM-F"}:
        return 4096
    return 1024


def _retrieval_minimum_samples(sequence_length: int) -> int:
    """Keep retrieval floors comparable in processed tokens, not examples."""
    return max(4096, (RETRIEVAL_MINIMUM_TOKENS + sequence_length - 1) // sequence_length)


def build_preflight_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "wave": "preflight",
            "lane": "language",
            "task": "small_language",
            "architecture": architecture,
            "scale": "10M",
            "target_parameter_budget": 10_000_000,
            "seed": 9101 + index,
            "data_seed": 19101 + index,
            "sequence_length": 128,
            "batch_size": 16,
            "num_supports": 1024,
            "top_k": 4,
            "minimum_tokens": 2_000_000,
            "target_seconds": TARGET_SECONDS["preflight"],
            "calibration": True,
        }
        for index, architecture in enumerate(("T0", "T-KAM-F", "T-MOE"))
    ]
    rows.append(
        {
            "wave": "preflight",
            "lane": "dynamics",
            "task": "switching_mackey_glass",
            "architecture": "T-KAM-ALT",
            "optimization": "alt_8_1",
            "scale": "2M",
            "target_parameter_budget": 1_000_000,
            "seed": 9104,
            "data_seed": 19104,
            "sequence_length": 64,
            "batch_size": 32,
            "num_supports": 256,
            "top_k": 4,
            "minimum_samples": 100_000,
            "target_seconds": TARGET_SECONDS["preflight"],
            "calibration": True,
        }
    )
    result = _finalize(rows)
    if len(result) != PREFLIGHT_ROWS:
        raise AssertionError("preflight row count changed")
    return result


def build_wave1_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    language_architectures = ("T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP")
    for architecture in language_architectures:
        for seed_tag, seed in enumerate((1101, 1102)):
            rows.append(
                {
                    "wave": "wave1",
                    "lane": "language",
                    "task": "small_language",
                    "architecture": architecture,
                    "optimization": "alt_8_1" if architecture == "T-KAM-ALT" else "vp_stop_gradient" if architecture == "T-KAM-VP" else "joint_sgd",
                    "scale": "10M",
                    "target_parameter_budget": 10_000_000,
                    "seed": seed,
                    "seed_tag": seed_tag,
                    "data_seed": 21_000 + seed,
                    "sequence_length": 128,
                    "batch_size": 16,
                    "num_supports": _language_supports(architecture),
                    "top_k": 4,
                    "minimum_tokens": 50_000_000,
                    "target_seconds": TARGET_SECONDS["wave1"],
                }
            )

    retrieval_design = (
        ("mqar", "T0", 128, 8, 4, "low"),
        ("mqar", "T-MEMTOK", 256, 16, 8, "medium"),
        ("mqar", "T-PKM", 512, 32, 8, "high"),
        ("associative_recall_distractors", "T-KAM-F", 256, 16, 4, "medium"),
        ("associative_recall_distractors", "T-KAM-L", 512, 32, 8, "high"),
        ("variable_copy", "T-KAM-F", 1024, 16, 4, "low"),
        ("variable_copy", "T-KAM-L", 128, 8, 4, "medium"),
    )
    for index, (task, architecture, sequence_length, bindings, queries, density) in enumerate(retrieval_design):
        rows.append(
            {
                "wave": "wave1",
                "lane": "retrieval",
                "task": task,
                "architecture": architecture,
                "scale": "10M",
                "target_parameter_budget": 10_000_000,
                "seed": 1201 + index,
                "data_seed": 22_201 + index,
                "sequence_length": sequence_length,
                "batch_size": 8 if sequence_length >= 512 else 16,
                "bindings": bindings,
                "queries": queries,
                "distractor_density": density,
                "num_supports": _retrieval_supports(architecture, sequence_length),
                "top_k": 4,
                "minimum_samples": _retrieval_minimum_samples(sequence_length),
                "target_seconds": TARGET_SECONDS["wave1"],
            }
        )

    dynamics_design = (
        ("switching_mackey_glass", "T0", "joint_sgd"),
        ("stable_switching_narma", "T-WIDE", "joint_sgd"),
        ("controlled_prototype", "T-KAM-F", "joint_sgd"),
        ("switching_mackey_glass", "T-KAM-L", "joint_sgd"),
        ("stable_switching_narma", "T-KAM-ALT", "alt_8_1"),
        ("controlled_prototype", "T-KAM-ALT", "alt_32_1"),
        ("lorenz63", "T-KAM-VP", "vp_stop_gradient"),
    )
    for index, (task, architecture, optimization) in enumerate(dynamics_design):
        rows.append(
            {
                "wave": "wave1",
                "lane": "dynamics",
                "task": task,
                "architecture": architecture,
                "optimization": optimization,
                "scale": "2M",
                "target_parameter_budget": 2_000_000,
                "seed": 1301 + index,
                "data_seed": 23_301 + index,
                "sequence_length": 64,
                "batch_size": 32,
                "num_supports": 256 if architecture.startswith("T-KAM") else 64,
                "top_k": 4,
                "minimum_samples": 100_000,
                "target_seconds": TARGET_SECONDS["wave1"],
            }
        )
    result = _finalize(rows)
    if len(result) != WAVE1_ROWS:
        raise AssertionError(f"wave1 row count changed: {len(result)}")
    return result


def _metric(row: dict[str, Any], name: str, default: float = float("inf")) -> float:
    value = row.get("metrics", {}).get(name, row.get(f"metric_{name}", row.get(name, default)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_architecture(rows: list[dict[str, Any]], candidates: set[str], fallback: str) -> str:
    valid = [row for row in rows if row.get("status") == "pass" and str(row.get("architecture")) in candidates]
    if not valid:
        return fallback
    grouped: dict[str, list[float]] = {}
    for row in valid:
        grouped.setdefault(str(row["architecture"]), []).append(_metric(row, "validation_loss"))
    return min(grouped, key=lambda name: sum(grouped[name]) / len(grouped[name]))


def build_wave2_rows(wave1_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conventional = _best_architecture(wave1_metrics, {"T-MEMTOK", "T-MOE", "T-PKM"}, "T-MOE")
    kam = _best_architecture(wave1_metrics, {"T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP"}, "T-KAM-F")
    architectures = ("T0", "T-WIDE", conventional, kam)
    rows: list[dict[str, Any]] = []
    for architecture in architectures:
        for seed_tag, seed in enumerate((2101, 2102, 2103)):
            rows.append(
                {
                    "wave": "wave2",
                    "lane": "language",
                    "task": "small_language",
                    "architecture": architecture,
                    "optimization": "alt_8_1" if architecture == "T-KAM-ALT" else "vp_stop_gradient" if architecture == "T-KAM-VP" else "joint_sgd",
                    "scale": "10M",
                    "target_parameter_budget": 10_000_000,
                    "seed": seed,
                    "seed_tag": seed_tag,
                    "data_seed": 31_000 + seed,
                    "sequence_length": 128,
                    "batch_size": 16,
                    "num_supports": _language_supports(architecture),
                    "top_k": 4,
                    "minimum_tokens": 150_000_000,
                    "target_seconds": TARGET_SECONDS["wave2"],
                    "promoted_from_wave1": True,
                }
            )
    for index, architecture in enumerate(architectures):
        rows.append(
            {
                "wave": "wave2",
                "lane": "dynamics_bundle",
                "task": "dynamics_bundle",
                "tasks": ["switching_mackey_glass", "stable_switching_narma", "controlled_prototype"],
                "architecture": architecture,
                "optimization": "alt_8_1" if architecture == "T-KAM-ALT" else "vp_stop_gradient" if architecture == "T-KAM-VP" else "joint_sgd",
                "scale": "2M",
                "target_parameter_budget": 2_000_000,
                "seed": 2201 + index,
                "seed_bundle": [2401, 2402, 2403, 2404, 2405],
                "data_seed": 32_201 + index,
                "sequence_length": 64,
                "batch_size": 32,
                "num_supports": 256 if architecture.startswith("T-KAM") else 64,
                "top_k": 4,
                "minimum_samples": 500_000,
                "target_seconds": TARGET_SECONDS["wave2"],
                "promoted_from_wave1": True,
            }
        )
    result = _finalize(rows)
    if len(result) != WAVE2_ROWS:
        raise AssertionError(f"wave2 row count changed: {len(result)}")
    return result


def build_wave3_rows(wave2_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conventional = _best_architecture(wave2_metrics, {"T-MEMTOK", "T-MOE", "T-PKM"}, "T-MOE")
    kam = _best_architecture(wave2_metrics, {"T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP"}, "T-KAM-F")
    architectures = ("T0", "T-WIDE", conventional, kam)
    rows: list[dict[str, Any]] = []
    for index, architecture in enumerate(architectures):
        rows.append(
            {
                "wave": "wave3",
                "lane": "language_replication",
                "task": "small_language",
                "architecture": architecture,
                "optimization": "alt_8_1" if architecture == "T-KAM-ALT" else "vp_stop_gradient" if architecture == "T-KAM-VP" else "joint_sgd",
                "scale": "10M",
                "target_parameter_budget": 10_000_000,
                "seed": 3101 + index,
                "seed_bundle": [3301, 3302, 3303],
                "data_seed": 43_101 + index,
                "sequence_length": 128,
                "batch_size": 16,
                "num_supports": _language_supports(architecture),
                "top_k": 4,
                "minimum_tokens_per_seed": 50_000_000,
                "target_seconds": TARGET_SECONDS["wave3"],
            }
        )
    for index, architecture in enumerate(architectures):
        rows.append(
            {
                "wave": "wave3",
                "lane": "adaptation",
                "task": "online_adaptation_bundle",
                "architecture": architecture,
                "adapter": "value_only" if architecture.startswith("T-KAM") else "rls",
                "seed": 3201 + index,
                "seed_bundle": [3401, 3402, 3403, 3404, 3405],
                "heldout_schedules_per_seed": 10,
                "tasks": ["mackey_glass_schedule", "narma_schedule", "prototype_schedule", "symbolic_schedule"],
                "num_supports": 256 if architecture.startswith("T-KAM") else 64,
                "top_k": 4,
                "target_seconds": TARGET_SECONDS["wave3"],
            }
        )
    result = _finalize(rows)
    if len(result) != WAVE3_ROWS:
        raise AssertionError(f"wave3 row count changed: {len(result)}")
    return result


def amend_wave1_timeout_rows(
    rows: list[dict[str, Any]], completed_row_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Amend only missing rows from the first Wave 1 timeout.

    Completed rows remain byte-for-byte identical. Missing rows receive a new
    content-addressed ID, an explicit provenance link to the superseded ID,
    registered memory sizes, and token-equivalent retrieval floors.
    """
    if len(rows) != WAVE1_ROWS:
        raise ValueError(f"Wave 1 repair requires {WAVE1_ROWS} rows, found {len(rows)}")
    amended: list[dict[str, Any]] = []
    repair: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row_id = str(row["row_id"])
        index = int(row["design_index"])
        if row_id in completed_row_ids:
            amended.append(row)
            continue
        if row.get("repair_revision") == 1:
            amended.append(row)
            repair.append(row)
            continue
        if index not in WAVE1_TIMEOUT_REPAIR_INDICES:
            raise ValueError(f"unexpected missing Wave 1 row at design_index={index}: {row_id}")
        row["supersedes_row_id"] = row_id
        row["repair_revision"] = 1
        row["repair_reason"] = "three_hour_timeout_from_unscaled_memory_or_example_budget"
        if row["lane"] == "language":
            row["num_supports"] = _language_supports(str(row["architecture"]))
        elif row["lane"] == "retrieval":
            sequence_length = int(row["sequence_length"])
            row["num_supports"] = _retrieval_supports(str(row["architecture"]), sequence_length)
            row["minimum_samples"] = _retrieval_minimum_samples(sequence_length)
            row["minimum_retrieval_tokens"] = RETRIEVAL_MINIMUM_TOKENS
        else:
            raise ValueError(f"timeout repair does not permit lane={row['lane']}")
        row.pop("row_id")
        row["row_id"] = _row_id(row)
        amended.append(row)
        repair.append(row)
    if len(repair) != len(WAVE1_TIMEOUT_REPAIR_INDICES):
        raise ValueError(f"expected {len(WAVE1_TIMEOUT_REPAIR_INDICES)} repair rows, found {len(repair)}")
    if len({row["row_id"] for row in amended}) != WAVE1_ROWS:
        raise ValueError("Wave 1 repair produced duplicate row IDs")
    return amended, repair


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    destination.write_text(content, encoding="utf-8")
    return {
        "path": str(destination),
        "rows": len(rows),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "CAMPAIGN",
    "PREFLIGHT_ROWS",
    "TARGET_SECONDS",
    "WAVE1_ROWS",
    "WAVE2_ROWS",
    "WAVE3_ROWS",
    "WAVE1_TIMEOUT_REPAIR_INDICES",
    "amend_wave1_timeout_rows",
    "build_preflight_rows",
    "build_wave1_rows",
    "build_wave2_rows",
    "build_wave3_rows",
    "read_jsonl",
    "write_manifest",
]
