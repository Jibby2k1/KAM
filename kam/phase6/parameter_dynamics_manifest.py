"""Manifest construction for the matched Phase 6.1 parameter-dynamics study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

CAMPAIGN = "phase6_parameter_dynamics_v1"
ARMS: dict[str, dict[str, Any]] = {
    "fixed_keys": {"freeze_fraction": 0.0, "optimization": "algebra_only", "role": "exact_structural_control"},
    "learned_joint_freeze50": {"freeze_fraction": 0.5, "optimization": "joint_sgd", "role": "freeze_timing_diagnostic"},
    "learned_joint_freeze80": {"freeze_fraction": 0.8, "optimization": "joint_sgd", "role": "primary_learned_arm"},
    "learned_alt8_freeze80": {"freeze_fraction": 0.8, "optimization": "alternating_8_1", "role": "optimizer_separation_arm"},
    "learned_joint_no_freeze": {"freeze_fraction": 1.0, "optimization": "joint_sgd", "role": "final_tuning_counterfactual"},
}
PILOT_SEEDS = (74_001, 74_002)
MAIN_SEEDS = tuple(range(74_101, 74_113))
MAIN_CHECKPOINTS = (0, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000)


def _row_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "p6dynamics_" + hashlib.sha256(payload).hexdigest()[:16]


def _base_row(*, stage: str, arm: str, seed: int, target_tokens: int) -> dict[str, Any]:
    arm_spec = ARMS[arm]
    checkpoints = [value for value in MAIN_CHECKPOINTS if value <= target_tokens]
    if target_tokens not in checkpoints:
        checkpoints.append(target_tokens)
    row: dict[str, Any] = {
        "campaign": CAMPAIGN, "stage": stage, "inferential": stage == "main", "arm": arm,
        "scientific_role": arm_spec["role"], "optimization": arm_spec["optimization"],
        "freeze_fraction": arm_spec["freeze_fraction"], "seed": seed, "training_seed": seed,
        "data_seed": 1_000_000 + seed, "pair_id": f"tinystories_v2_128mib:{seed}",
        "corpus_id": "tinystories_v2_128mib",
        "corpus_train_path": "data/phase6_confirmation/TinyStoriesV2-GPT4-train.128MiB.txt",
        "corpus_validation_path": "data/phase6_confirmation/TinyStories-valid.validation.txt",
        "corpus_test_path": "data/phase6_confirmation/TinyStories-valid.test.txt",
        "d_model": 104, "n_heads": 8, "n_layers": 8, "d_ff": 416, "vocab_size": 256,
        "sequence_length": 128, "batch_size": 16, "num_supports": 1024, "top_k": 4,
        "expert_mode": "low_rank", "expert_rank": 4, "target_parameter_budget": 10_000_000,
        "parameter_tolerance_fraction": 0.01, "target_tokens": target_tokens,
        "validation_token_checkpoints": sorted(checkpoints), "precision": "bf16",
        "save_snapshots": True, "trace_schema_version": 1, "preregistered": True,
    }
    row["row_id"] = _row_id(row)
    return row


def build_parameter_dynamics_rows(stage: str) -> list[dict[str, Any]]:
    if stage == "pilot":
        seeds, target_tokens = PILOT_SEEDS, 5_000_000
    elif stage == "main":
        seeds, target_tokens = MAIN_SEEDS, 50_000_000
    else:
        raise ValueError("stage must be pilot or main")
    rows = [_base_row(stage=stage, arm=arm, seed=seed, target_tokens=target_tokens) for seed in seeds for arm in ARMS]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise AssertionError("parameter-dynamics manifest contains duplicate row IDs")
    return rows


def write_manifest(path: str | Path, stage: str, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows if rows is not None else build_parameter_dynamics_rows(stage))
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized).encode()
    destination.write_bytes(payload)
    return {"campaign": CAMPAIGN, "stage": stage, "path": str(destination), "rows": len(materialized),
            "sha256": hashlib.sha256(payload).hexdigest(), "arms": list(ARMS),
            "paired_seeds": len(PILOT_SEEDS if stage == "pilot" else MAIN_SEEDS)}


__all__ = ["ARMS", "CAMPAIGN", "MAIN_CHECKPOINTS", "MAIN_SEEDS", "PILOT_SEEDS", "build_parameter_dynamics_rows", "write_manifest"]
