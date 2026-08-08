from __future__ import annotations

import torch

from kam.phase6.behavioral_atlas_instrumentation import _strict_fp32_logits
from kam.phase6.behavioral_atlas_manifest import build_behavioral_atlas_rows
from kam.phase6.behavioral_atlas_permutation_audit import (
    _operational_logits,
    _permutation_measurement,
    build_audit_specs,
)
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model
from kam.phase6.parameter_trace import state_hash


def _result(row_id: str, arm: str, *, semantic: float, top1: float, kl: float) -> dict:
    return {
        "row_id": row_id,
        "arm": arm,
        "seed": int(row_id.rsplit("_", 1)[-1]),
        "matched_key_expert_permutation": {
            "passed": semantic <= 2e-5,
            "max_abs_logit_difference": semantic,
            "operational_top1_flip_rate": top1,
            "operational_predictive_kl": kl,
            "operational_within_expected_precision_tolerance": top1 <= 2e-2 and kl <= 1e-3,
        },
    }


def test_audit_selection_is_balanced_bounded_and_keeps_strict_failures() -> None:
    rows = []
    for arm in ("arm_a", "arm_b"):
        rows.extend([
            _result(f"{arm}_1", arm, semantic=0.0, top1=0.005, kl=0.0002),
            _result(f"{arm}_2", arm, semantic=0.0, top1=0.019, kl=0.0009),
            _result(f"{arm}_3", arm, semantic=0.0, top1=0.021, kl=0.0008),
            _result(f"{arm}_4", arm, semantic=0.0, top1=0.010, kl=0.0030),
        ])
    rows.append(_result("arm_a_5", "arm_a", semantic=3e-5, top1=0.005, kl=0.0002))
    specs = build_audit_specs(rows)
    by_source = {spec["source_row_id"]: spec for spec in specs}
    assert set(by_source) == {
        "arm_a_2", "arm_a_3", "arm_a_4", "arm_a_5", "arm_b_2", "arm_b_3", "arm_b_4"
    }
    assert "strict_fp32_failure" in by_source["arm_a_5"]["selection_roles"]
    assert "bf16_closest_failure" in by_source["arm_a_3"]["selection_roles"]
    assert "bf16_worst_failure" in by_source["arm_a_4"]["selection_roles"]
    assert "bf16_closest_pass" in by_source["arm_a_2"]["selection_roles"]
    assert all(not spec["inferential"] and not spec["retraining"] for spec in specs)


def test_permutation_measurement_restores_checkpoint_and_preserves_function() -> None:
    row = next(
        row.copy()
        for row in build_behavioral_atlas_rows("stage0")
        if row["arm"] == "learned_joint_freeze80" and row["profile_kind"] is None
    )
    row.update({
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 32,
        "vocab_size": 16,
        "sequence_length": 8,
        "num_supports": 8,
        "top_k": 2,
        "expert_rank": 4,
        "precision": "fp32",
    })
    torch.manual_seed(7)
    model = build_behavioral_atlas_model(row).eval()
    sample = torch.randint(0, 16, (4, 8))
    before = state_hash(model)
    with torch.inference_mode():
        semantic = _strict_fp32_logits(model, sample, torch.device("cpu"))
        operational = _operational_logits(model, sample, torch.device("cpu"), "fp32")
    result = _permutation_measurement(
        model,
        sample,
        device=torch.device("cpu"),
        precision="fp32",
        seed=19,
        layer_indices=[0, 1],
        semantic_baseline=semantic,
        operational_baseline=operational,
    )
    assert result["semantic"]["passed"]
    assert result["operational"]["passed"]
    assert state_hash(model) == before
