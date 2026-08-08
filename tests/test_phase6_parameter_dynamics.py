from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kam.phase6.parameter_dynamics_analysis import analyze_parameter_dynamics
from kam.phase6.parameter_dynamics_manifest import ARMS, build_parameter_dynamics_rows
from kam.phase6.parameter_dynamics_runner import (
    _seed_everything,
    build_parameter_dynamics_model,
    run_parameter_dynamics_row,
)
from kam.phase6.parameter_trace import GROUPS, grouped_named_parameters, state_hash
from kam.phase6.parameter_dynamics_statistics import holm_adjust, paired_log_comparison


def test_parameter_dynamics_manifest_is_fixed_and_paired() -> None:
    pilot = build_parameter_dynamics_rows("pilot")
    main = build_parameter_dynamics_rows("main")
    assert len(pilot) == 10
    assert len(main) == 60
    assert len({row["row_id"] for row in main}) == 60
    for seed in range(74_101, 74_113):
        rows = [row for row in main if row["seed"] == seed]
        assert {row["arm"] for row in rows} == set(ARMS)
        assert len({row["data_seed"] for row in rows}) == 1
        assert len({row["target_tokens"] for row in rows}) == 1


def test_all_arms_have_identical_initial_state_and_shape() -> None:
    rows = [row for row in build_parameter_dynamics_rows("pilot") if row["seed"] == 74_001]
    hashes = set()
    shapes = set()
    for row in rows:
        _seed_everything(int(row["seed"]))
        model = build_parameter_dynamics_model(row)
        hashes.add(state_hash(model))
        shapes.add((model.config.d_model, model.config.n_layers, tuple(model.memory_layers[0].keys.shape)))
        assert set(grouped_named_parameters(model)) == set(GROUPS)
    assert len(hashes) == 1
    assert shapes == {(104, 8, (1024, 104))}


def _tiny_row(tmp_path: Path, arm: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    row = next(row for row in build_parameter_dynamics_rows("pilot") if row["seed"] == 74_001 and row["arm"] == arm)
    for name, byte in (("train", b"a"), ("validation", b"b"), ("test", b"c")):
        path = tmp_path / f"{name}.txt"
        path.write_bytes(byte * 6000)
        row[f"corpus_{name}_path"] = str(path)
    row.update(
        {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 64,
            "sequence_length": 8,
            "batch_size": 2,
            "num_supports": 32,
            "top_k": 2,
            "expert_rank": 4,
            "target_tokens": 128,
            "validation_token_checkpoints": [0, 32, 64, 96, 128],
            "precision": "fp32",
            "save_snapshots": False,
        }
    )
    _seed_everything(int(row["seed"]))
    row["target_parameter_budget"] = sum(parameter.numel() for parameter in build_parameter_dynamics_model(row).parameters())
    row["parameter_tolerance_fraction"] = 0.0
    return row


def test_event_ordered_freeze_and_exact_postfreeze_integrity(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path, "learned_joint_freeze50")
    result = run_parameter_dynamics_row(row, device="cpu", output_root=tmp_path / "run")
    phases = [point["phase"] for point in result["traces"]]
    event = phases.index("freeze_event")
    assert all(phase != "post_freeze" for phase in phases[:event])
    assert all(phase == "post_freeze" for phase in phases[event + 1 :])
    assert result["postfreeze_key_hash_unchanged"]
    assert result["postfreeze_relative_l2_drift"] == 0.0
    assert not result["postfreeze_key_grad_observed"]
    assert result["geometry_steps"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_fused_optimizer_smoke(tmp_path: Path) -> None:
    row = _tiny_row(tmp_path, "learned_alt8_freeze80")
    row["precision"] = "bf16"
    result = run_parameter_dynamics_row(row, device="cuda", output_root=tmp_path / "gpu_run")
    assert result["execution"]["fused_adamw"]
    assert result["execution"]["bf16_supported"]
    assert result["execution"]["tf32"]
    assert result["peak_vram_bytes"] > 0
    assert result["status"] == "pass"


def test_tiny_end_to_end_report_builds_all_figures(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    rows = [_tiny_row(tmp_path / arm, arm) for arm in ("fixed_keys", "learned_joint_freeze50")]
    for row in rows:
        run_parameter_dynamics_row(row, device="cpu", output_root=run_root)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = analyze_parameter_dynamics(run_root, tmp_path / "report", manifest)
    assert summary["audit"]["passed"]
    assert len(summary["figures"]) == 16
    assert all(Path(path).is_file() for path in summary["figures"])
    assert (tmp_path / "report" / "PARAMETER_DYNAMICS_REPORT.md").is_file()


def test_paired_statistics_use_seed_identity_and_holm_adjustment() -> None:
    results = []
    for seed in range(12):
        results.append({"seed": seed, "arm": "learned_joint_freeze80", "test_loss": 0.9,
                        "traces": [{"checkpoint_target_tokens": 40_000_000, "validation_loss": 0.8}]})
        results.append({"seed": seed, "arm": "fixed_keys", "test_loss": 1.0,
                        "traces": [{"checkpoint_target_tokens": 40_000_000, "validation_loss": 1.0}]})
    comparison = paired_log_comparison(results, "learned_joint_freeze80", "fixed_keys",
                                       metric="validation_loss", checkpoint=40_000_000)
    assert comparison["paired_seeds"] == 12
    assert comparison["geometric_relative_change"] < 0
    assert comparison["paired_sign_flip_p"] < 0.05
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
