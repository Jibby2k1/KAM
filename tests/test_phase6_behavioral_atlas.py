from __future__ import annotations

import json
from pathlib import Path

import torch

from kam.phase6.behavioral_atlas_analysis import analyze_behavioral_atlas, forecast_manifest
from kam.phase6.behavioral_atlas_instrumentation import (
    WindowDynamicsAccumulator,
    build_anchor_bank,
    evaluate_anchor_behavior,
    matched_key_expert_permutation_check,
)
from kam.phase6.behavioral_atlas_manifest import ARMS, build_behavioral_atlas_rows
from kam.phase6.behavioral_atlas_runner import (
    build_behavioral_atlas_model,
    run_behavioral_atlas_row,
)
from kam.phase6.parameter_dynamics_runner import _seed_everything
from kam.phase6.parameter_trace import state_hash


def test_behavioral_atlas_manifest_counts_and_executable_labels() -> None:
    profile = build_behavioral_atlas_rows("l4_profile")
    stage0 = build_behavioral_atlas_rows("stage0")
    assert len(profile) == 1
    assert len(stage0) == 24
    assert len({row["row_id"] for row in stage0}) == 24
    functional = [row for row in stage0 if row["profile_kind"] is None]
    assert len(functional) == 18
    for seed in (76_001, 76_002, 76_003):
        assert {row["arm"] for row in functional if row["seed"] == seed} == set(ARMS)
    assert all("sgd" not in row["optimization"] for row in stage0)


def test_stage1_manifest_matches_preregistered_paired_design() -> None:
    rows = build_behavioral_atlas_rows("stage1_core_lifecycle")
    assert len(rows) == 168
    assert len({row["row_id"] for row in rows}) == 168
    assert all(row["inferential"] and row["target_tokens"] == 50_000_000 for row in rows)
    assert all(row["anchor_token_states"] == 16_384 for row in rows)
    primary_only = [row for row in rows if row["seed"] >= 76_113]
    assert len(primary_only) == 18 * 4
    assert all(row["arm"] in {
        "fixed_keys",
        "learned_joint_adamw_freeze80",
        "learned_joint_adamw_no_freeze",
        "learned_alt8_adamw_freeze80",
    } for row in primary_only)
    decay = [row for row in rows if row["arm"] == "learned_joint_adamw_cosine_geometry_decay"]
    assert len(decay) == 12
    assert all(row["geometry_lr_schedule"] == "cosine" and row["freeze_fraction"] == 1.0 for row in decay)


def test_canonical_kam_arms_share_initial_state_within_seed() -> None:
    rows = [row for row in build_behavioral_atlas_rows("stage0") if row["seed"] == 76_001 and row["profile_kind"] is None and row["architecture"] == "canonical_kam"]
    hashes = set()
    for row in rows:
        _seed_everything(int(row["seed"]))
        hashes.add(state_hash(build_behavioral_atlas_model(row)))
    assert len(hashes) == 1


def _tiny_row(tmp_path: Path, arm: str = "learned_joint_freeze80") -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    row = next(row.copy() for row in build_behavioral_atlas_rows("stage0") if row["arm"] == arm and row["profile_kind"] is None)
    for name, byte in (("train", b"a"), ("validation", b"b"), ("test", b"c")):
        path = tmp_path / f"{name}.txt"
        path.write_bytes(byte * 6000)
        row[f"corpus_{name}_path"] = str(path)
    row.update({
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 32,
        "sequence_length": 8,
        "batch_size": 2,
        "num_supports": 8,
        "top_k": 2,
        "expert_rank": 4,
        "target_parameter_budget": None,
        "target_tokens": 128,
        "validation_token_checkpoints": [0, 32, 64, 96, 128],
        "precision": "fp32",
        "anchor_token_states": 64,
        "anchor_batch_size": 2,
        "window_sample_stride": 2,
        "save_snapshots": False,
        "compile_training": False,
    })
    row["row_id"] = f"tiny_{arm}_{tmp_path.name}"
    return row


def test_anchor_bank_is_immutable_and_behavior_decomposes_routing(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path)
    tokens = torch.arange(512, dtype=torch.long) % 16
    first = build_anchor_bank(tokens, (0, 512), sequence_length=8, token_states=64, seed=7)
    second = build_anchor_bank(tokens, (0, 512), sequence_length=8, token_states=64, seed=7)
    changed = build_anchor_bank(tokens, (0, 512), sequence_length=8, token_states=64, seed=8)
    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256
    _seed_everything(int(row["seed"]))
    model = build_behavioral_atlas_model(row)
    metrics, reference = evaluate_anchor_behavior(model, first, batch_size=2, device=torch.device("cpu"), precision="fp32", top_k=2)
    assert metrics["routing_decomposition"]["states"]["Q0_K0"]["jaccard_to_Q0_K0"] == 1.0
    assert metrics["routing_decomposition"]["states"]["Qt_Kt"]["jaccard_to_Q0_K0"] == 1.0
    assert 0 <= metrics["routing_decomposition"]["states"]["Qt_Kt"]["dead_support_fraction"] <= 1
    assert len(reference.queries) == 2


def test_matched_key_expert_permutation_is_function_symmetry(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path)
    _seed_everything(int(row["seed"]))
    model = build_behavioral_atlas_model(row)
    inputs = torch.randint(0, 16, (4, 8))
    result = matched_key_expert_permutation_check(model, inputs, device=torch.device("cpu"), precision="fp32", seed=9)
    assert result["applicable"]
    assert result["semantic_precision"] == "fp32"
    assert result["passed"]
    assert result["operational_within_expected_precision_tolerance"]
    assert result["operational_top1_flip_rate"] <= result["operational_top1_flip_tolerance"]
    assert result["operational_predictive_kl"] <= result["operational_predictive_kl_tolerance"]
    assert result["max_abs_logit_difference"] <= result["tolerance"]


def test_window_accumulator_reports_distributional_statistics() -> None:
    accumulator = WindowDynamicsAccumulator()
    update = {group: {"optimizer_update_l2_norm": 0.1, "update_to_weight_ratio": 0.01} for group in accumulator.values}
    for scale in (1.0, 2.0, 3.0):
        gradients = {group: scale for group in accumulator.values}
        accumulator.observe(raw_gradients=gradients, clipped_gradients=gradients, updates=update)
    summary = accumulator.summarize()
    assert summary["memory_keys"]["raw_gradient_l2_norm"]["samples"] == 3
    assert summary["memory_keys"]["raw_gradient_l2_norm"]["median"] == 2.0
    assert summary["memory_keys"]["raw_gradient_l2_norm"]["norm_snr"] is not None


def test_tiny_end_to_end_stage0_records_provenance_behavior_and_freeze(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path)
    result = run_behavioral_atlas_row(row, device="cpu", output_root=tmp_path / "run")
    assert result["status"] == "pass"
    assert result["optimizer_provenance"]["effective_optimizer_class"] == "AdamW"
    assert result["optimizer_provenance"]["label_matches_executable"]
    assert result["anchor_token_states_resolved"] == 64
    assert result["sample_order_sha256"]
    assert result["matched_key_expert_permutation"]["passed"]
    assert result["postfreeze_key_hash_unchanged"]
    assert result["postfreeze_relative_l2_drift"] == 0.0
    assert any(point["behavior"] is not None for point in result["traces"])
    assert any(point["window_dynamics"]["memory_keys"] for point in result["traces"][1:])


def test_cosine_geometry_schedule_decays_without_freezing(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path, "learned_joint_adamw")
    row["geometry_lr_schedule"] = "cosine"
    result = run_behavioral_atlas_row(row, device="cpu", output_root=tmp_path / "run_cosine")
    learning_rates = [point["learning_rates"]["geometry"] for point in result["traces"]]
    assert learning_rates[0] > learning_rates[-1]
    assert result["freeze_tokens"] is None
    assert result["final_geometry_learning_rate"] < row["geometry_lr"]
    assert result["optimizer_provenance"]["effective_schedule"].endswith("geometry_lr_cosine")


def test_t0_control_runs_without_memory(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path, "T0")
    result = run_behavioral_atlas_row(row, device="cpu", output_root=tmp_path / "run_t0")
    assert result["status"] == "pass"
    assert result["geometry_steps"] == 0
    assert result["matched_key_expert_permutation"] == {
        "applicable": False,
        "passed": True,
        "max_abs_logit_difference": 0.0,
        "mean_abs_logit_difference": 0.0,
    }
    assert result["traces"][-1]["behavior"]["routing_decomposition"]["states"] == {}


def test_forecast_and_tiny_report_build(tmp_path: Path) -> None:
    rows = [_tiny_row(tmp_path / arm, arm) for arm in ("fixed_keys", "learned_joint_freeze80")]
    run_root = tmp_path / "run"
    for row in rows:
        run_behavioral_atlas_row(row, device="cpu", output_root=run_root)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    forecast = forecast_manifest(rows)
    assert forecast["rows"] == 2
    assert forecast["required_with_25_percent_headroom_gib"] > 0
    summary = analyze_behavioral_atlas(run_root, tmp_path / "report", manifest)
    assert summary["audit"]["passed"]
    assert summary["decision"] == "STAGE0_BLOCKED"
    assert len(summary["figures"]) == 16
    assert all(Path(path).is_file() for path in summary["figures"])
    assert (tmp_path / "report" / "BEHAVIORAL_ATLAS_STAGE0_REPORT.md").is_file()
    report_text = (tmp_path / "report" / "BEHAVIORAL_ATLAS_STAGE0_REPORT.md").read_text()
    assert "## Descriptive results" in report_text
    assert "Permutation precision diagnostics" in report_text
    assert len(summary["descriptive_rows"]) == 2
