from __future__ import annotations

import json
from pathlib import Path

import torch

from kam.phase6.behavioral_atlas_instrumentation import _strict_fp32_logits
from kam.phase6.behavioral_atlas_manifest import build_behavioral_atlas_rows
from kam.phase6.behavioral_atlas_permutation_audit import _operational_logits
from kam.phase6.behavioral_atlas_permutation_localization import (
    _fixed_permutations,
    _subset_measurement,
    write_manifest,
)
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model
from kam.phase6.parameter_trace import state_hash


def test_localization_manifest_is_exactly_six_gpu_and_two_cpu_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "permutation_audit_summary.json").write_text(json.dumps({
        "reproduced_semantic_failures": ["failure_b", "failure_a"],
    }))
    destination = tmp_path / "manifest.jsonl"
    summary = write_manifest(audit, destination)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert summary["rows"] == 8
    assert summary["gpu_rows"] == 6
    assert summary["cpu_rows"] == 2
    assert {row["source_row_id"] for row in rows} == {"failure_a", "failure_b"}
    assert all(not row["inferential"] and not row["retraining"] for row in rows)


def test_fixed_subset_permutation_restores_model_and_preserves_cpu_function() -> None:
    row = next(
        row.copy()
        for row in build_behavioral_atlas_rows("stage0")
        if row["arm"] == "learned_joint_freeze80" and row["profile_kind"] is None
    )
    row.update({
        "d_model": 16, "n_heads": 4, "n_layers": 2, "d_ff": 32, "vocab_size": 16,
        "sequence_length": 8, "num_supports": 8, "top_k": 2, "expert_rank": 4,
    })
    torch.manual_seed(11)
    model = build_behavioral_atlas_model(row).eval()
    sample = torch.randint(0, 16, (4, 8))
    before = state_hash(model)
    with torch.inference_mode():
        semantic = _strict_fp32_logits(model, sample, torch.device("cpu"))
        operational = _operational_logits(model, sample, torch.device("cpu"), "fp32")
    permutations = _fixed_permutations(model, 99)
    result = _subset_measurement(
        model, sample, device=torch.device("cpu"), precision="fp32",
        permutations=permutations, layer_indices=(1,),
        semantic_baseline=semantic, operational_baseline=operational,
    )
    assert result["semantic"]["passed"]
    assert result["operational"]["passed"]
    assert state_hash(model) == before
