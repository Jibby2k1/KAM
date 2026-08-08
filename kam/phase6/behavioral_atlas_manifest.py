"""Deterministic manifests for the Phase 6.2 Stage 0 behavioral atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

CAMPAIGN = "phase6_behavioral_atlas_v2"
STAGE0_SEEDS = (76_001, 76_002, 76_003)
STAGE0_TARGET_TOKENS = 2_000_000
STAGE0_CHECKPOINTS = (0, 250_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000, 1_600_000, 1_800_000, 2_000_000)
STAGE1_PRIMARY_SEEDS = tuple(range(76_101, 76_131))
STAGE1_SECONDARY_SEEDS = tuple(range(76_101, 76_113))
STAGE1_TARGET_TOKENS = 50_000_000
STAGE1_CHECKPOINTS = (0, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000)

ARMS: dict[str, dict[str, Any]] = {
    "T0": {
        "architecture": "T0",
        "identity_group": "dense_control",
        "geometry_trainable": False,
        "freeze_fraction": 0.0,
        "optimization": "dense_adamw",
        "alternating_ratio": None,
    },
    "fixed_keys": {
        "architecture": "canonical_kam",
        "identity_group": "canonical_kam",
        "geometry_trainable": False,
        "freeze_fraction": 0.0,
        "optimization": "algebra_only_adamw",
        "alternating_ratio": None,
    },
    "learned_joint_adamw": {
        "architecture": "canonical_kam",
        "identity_group": "canonical_kam",
        "geometry_trainable": True,
        "freeze_fraction": 1.0,
        "optimization": "joint_adamw",
        "alternating_ratio": None,
    },
    "learned_joint_freeze80": {
        "architecture": "canonical_kam",
        "identity_group": "canonical_kam",
        "geometry_trainable": True,
        "freeze_fraction": 0.8,
        "optimization": "joint_adamw",
        "alternating_ratio": None,
    },
    "learned_alt8_freeze80": {
        "architecture": "canonical_kam",
        "identity_group": "canonical_kam",
        "geometry_trainable": True,
        "freeze_fraction": 0.8,
        "optimization": "alternating_adamw",
        "alternating_ratio": 8,
    },
    "learned_alt32_freeze80": {
        "architecture": "canonical_kam",
        "identity_group": "canonical_kam",
        "geometry_trainable": True,
        "freeze_fraction": 0.8,
        "optimization": "alternating_adamw",
        "alternating_ratio": 32,
    },
}

STAGE1_ARMS: dict[str, dict[str, Any]] = {
    "fixed_keys": ARMS["fixed_keys"],
    "learned_joint_adamw_freeze80": ARMS["learned_joint_freeze80"],
    "learned_joint_adamw_no_freeze": ARMS["learned_joint_adamw"],
    "learned_alt8_adamw_freeze80": ARMS["learned_alt8_freeze80"],
    "learned_joint_adamw_freeze25": {**ARMS["learned_joint_freeze80"], "freeze_fraction": 0.25},
    "learned_joint_adamw_freeze50": {**ARMS["learned_joint_freeze80"], "freeze_fraction": 0.50},
    "learned_alt32_adamw_freeze80": ARMS["learned_alt32_freeze80"],
    "learned_joint_adamw_cosine_geometry_decay": {
        **ARMS["learned_joint_adamw"],
        "geometry_lr_schedule": "cosine",
    },
}

STAGE1_PRIMARY_ARMS = (
    "fixed_keys",
    "learned_joint_adamw_freeze80",
    "learned_joint_adamw_no_freeze",
    "learned_alt8_adamw_freeze80",
)
STAGE1_SECONDARY_ARMS = (
    "learned_joint_adamw_freeze25",
    "learned_joint_adamw_freeze50",
    "learned_alt32_adamw_freeze80",
    "learned_joint_adamw_cosine_geometry_decay",
)

PROFILE_KINDS: tuple[dict[str, Any], ...] = (
    {"profile_kind": "trace_off", "trace_level": "off", "anchor_token_states": 0, "compile_training": False},
    {"profile_kind": "standard_trace", "trace_level": "standard", "anchor_token_states": 8192, "compile_training": False},
    {"profile_kind": "doubled_anchor", "trace_level": "standard", "anchor_token_states": 16384, "compile_training": False},
    {"profile_kind": "repeatability_a", "trace_level": "standard", "anchor_token_states": 8192, "compile_training": False},
    {"profile_kind": "repeatability_b", "trace_level": "standard", "anchor_token_states": 8192, "compile_training": False},
    {"profile_kind": "compile_candidate", "trace_level": "standard", "anchor_token_states": 8192, "compile_training": True},
)


def _row_id(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "p6atlas_" + hashlib.sha256(payload).hexdigest()[:16]


def _base_row(
    *,
    stage: str,
    arm: str,
    seed: int,
    target_tokens: int,
    profile_kind: str | None = None,
    trace_level: str = "standard",
    anchor_token_states: int = 8192,
    compile_training: bool = False,
) -> dict[str, Any]:
    spec = (STAGE1_ARMS if stage == "stage1_core_lifecycle" else ARMS)[arm]
    if stage == "stage0":
        checkpoints = [value for value in STAGE0_CHECKPOINTS if value <= target_tokens]
    elif stage == "stage1_core_lifecycle":
        checkpoints = [value for value in STAGE1_CHECKPOINTS if value <= target_tokens]
    else:
        checkpoints = [0, target_tokens // 2, target_tokens]
    checkpoints = sorted(set(checkpoints + [0, target_tokens]))
    row: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "stage": stage,
        "inferential": stage == "stage1_core_lifecycle",
        "arm": arm,
        "profile_kind": profile_kind,
        **spec,
        "seed": seed,
        "training_seed": seed,
        "data_seed": 2_000_000 + seed,
        "anchor_seed": 7_600_000 + seed,
        "pair_id": f"tinystories_v2_128mib:{seed}",
        "corpus_id": "tinystories_v2_128mib",
        "corpus_train_path": "data/phase6_confirmation/TinyStoriesV2-GPT4-train.128MiB.txt",
        "corpus_validation_path": "data/phase6_confirmation/TinyStories-valid.validation.txt",
        "corpus_test_path": "data/phase6_confirmation/TinyStories-valid.test.txt",
        "d_model": 104,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 416,
        "vocab_size": 256,
        "sequence_length": 128,
        "batch_size": 16,
        "num_supports": 1024,
        "top_k": 4,
        "router_metric": "dot",
        "router_temperature": 1.0,
        "expert_mode": "low_rank",
        "expert_rank": 4,
        "target_parameter_budget": None,
        "parameter_tolerance_fraction": 0.01,
        "target_tokens": int(target_tokens),
        "validation_token_checkpoints": checkpoints,
        "precision": "bf16",
        "algebra_lr": 3e-4,
        "geometry_lr": 3e-5,
        "algebra_weight_decay": 0.1,
        "geometry_weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "trace_schema_version": 2,
        "trace_level": trace_level,
        "anchor_token_states": int(anchor_token_states),
        "anchor_batch_size": 16,
        "window_sample_stride": 64,
        "save_snapshots": (stage == "stage0" and profile_kind is None) or stage == "stage1_core_lifecycle",
        "compile_training": bool(compile_training),
        "compile_mode": "default",
        "compile_cudagraphs": False,
        "preregistered": True,
        "scientific_role": (
            "inferential_stage1_core_lifecycle"
            if stage == "stage1_core_lifecycle"
            else "noninferential_stage0_instrumentation"
        ),
    }
    row["row_id"] = _row_id(row)
    return row


def build_behavioral_atlas_rows(stage: str) -> list[dict[str, Any]]:
    if stage in {"l4_profile", "l4_profile_r2", "l4_profile_r3"}:
        rows = [
            _base_row(
                stage=stage,
                arm="learned_joint_freeze80",
                seed=76_000,
                target_tokens=250_000,
                profile_kind="bounded_l4_profile",
                trace_level="standard",
                anchor_token_states=8192,
            )
        ]
    elif stage == "stage0":
        rows = [
            _base_row(stage=stage, arm=arm, seed=seed, target_tokens=STAGE0_TARGET_TOKENS)
            for seed in STAGE0_SEEDS
            for arm in ARMS
        ]
        rows.extend(
            _base_row(
                stage=stage,
                arm="learned_joint_freeze80",
                seed=76_001,
                target_tokens=STAGE0_TARGET_TOKENS,
                **profile,
            )
            for profile in PROFILE_KINDS
        )
    elif stage == "stage1_core_lifecycle":
        rows = [
            _base_row(
                stage=stage, arm=arm, seed=seed, target_tokens=STAGE1_TARGET_TOKENS,
                anchor_token_states=16_384,
            )
            for seed in STAGE1_PRIMARY_SEEDS
            for arm in STAGE1_PRIMARY_ARMS
        ]
        rows.extend(
            _base_row(
                stage=stage, arm=arm, seed=seed, target_tokens=STAGE1_TARGET_TOKENS,
                anchor_token_states=16_384,
            )
            for seed in STAGE1_SECONDARY_SEEDS
            for arm in STAGE1_SECONDARY_ARMS
        )
    else:
        raise ValueError("stage must be l4_profile, l4_profile_r2, l4_profile_r3, stage0, or stage1_core_lifecycle")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise AssertionError("behavioral-atlas manifest contains duplicate row IDs")
    return rows


def write_manifest(path: str | Path, stage: str, rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows if rows is not None else build_behavioral_atlas_rows(stage))
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized).encode()
    destination.write_bytes(payload)
    return {
        "campaign": CAMPAIGN,
        "stage": stage,
        "path": str(destination),
        "rows": len(materialized),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "arms": sorted({str(row["arm"]) for row in materialized}),
        "inferential": bool(materialized and materialized[0].get("inferential")),
    }


__all__ = [
    "ARMS",
    "CAMPAIGN",
    "PROFILE_KINDS",
    "STAGE0_CHECKPOINTS",
    "STAGE0_SEEDS",
    "STAGE1_ARMS",
    "STAGE1_CHECKPOINTS",
    "STAGE1_PRIMARY_ARMS",
    "STAGE1_PRIMARY_SEEDS",
    "STAGE1_SECONDARY_ARMS",
    "STAGE1_SECONDARY_SEEDS",
    "build_behavioral_atlas_rows",
    "write_manifest",
]
