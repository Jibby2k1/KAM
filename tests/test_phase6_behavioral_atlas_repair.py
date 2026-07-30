from __future__ import annotations

from pathlib import Path

from kam.phase6.behavioral_atlas_repair import build_repair_rows
from kam.phase6.behavioral_atlas_stage1_analysis import PRIMARY_COMPARISONS, paired_comparisons


def test_repair_manifest_is_bounded_and_revisioned() -> None:
    rows = build_repair_rows()
    assert len(rows) == 2
    assert len({row["row_id"] for row in rows}) == 2
    anchor = next(row for row in rows if row["repair_kind"] == "anchor_checkpoint_reevaluation")
    compiled = next(row for row in rows if row["repair_kind"] == "compile_candidate_no_cudagraph")
    assert anchor["standard_anchor_token_states"] == 16_384
    assert anchor["doubled_anchor_token_states"] == 32_768
    assert compiled["compile_training"]
    assert compiled["compile_mode"] == "default"
    assert compiled["compile_cudagraphs"] is False
    assert compiled["supersedes_row_id"]


def test_repair_manifest_can_resolve_original_immutable_stage0_ids() -> None:
    manifest = Path("configs/phase6/behavioral_atlas_v2_stage0_manifest.jsonl")
    rows = build_repair_rows(manifest)
    anchor = next(row for row in rows if row["repair_kind"] == "anchor_checkpoint_reevaluation")
    compiled = next(row for row in rows if row["repair_kind"] == "compile_candidate_no_cudagraph")
    assert anchor["source_row_id"] == "p6atlas_493d613d96cdefa3"
    assert compiled["supersedes_row_id"] == "p6atlas_30ed78b738bcf849"


def test_paired_randomization_and_holm_are_seed_paired() -> None:
    arms = {arm for comparison in PRIMARY_COMPARISONS for arm in comparison}
    rows = []
    for seed in range(30):
        offsets = {
            "fixed_keys": 0.10,
            "learned_joint_adamw_no_freeze": 0.06,
            "learned_joint_adamw_freeze80": 0.02,
            "learned_alt8_adamw_freeze80": 0.00,
        }
        for arm in arms:
            rows.append({"arm": arm, "seed": seed, "test_loss": 2.0 + offsets[arm] + seed * 1e-4})
    comparisons = paired_comparisons(rows, PRIMARY_COMPARISONS, "primary")
    assert len(comparisons) == 4
    assert all(row["n"] == 30 for row in comparisons)
    assert all(0 <= row["holm_adjusted_p"] <= 1 for row in comparisons)
    assert all(row["mean_log_loss_ratio"] < 0 for row in comparisons)
    assert all(row["geometric_relative_change"] < 0 for row in comparisons)
    assert all(row["win_rate_first_lower_loss"] == 1.0 for row in comparisons)
