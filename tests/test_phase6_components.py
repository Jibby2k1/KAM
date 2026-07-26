from __future__ import annotations

import torch

from kam.data.phase6 import controlled_symbolic_regimes, language_batches, load_text_tokens, mqar, variable_copy
from kam.memory import ApproximateTopKRouter, SparseMemoryConfig, SparseSeparableMemory, fixed_data_sample, kmeans
from kam.optimization import GeometryTrustRegion, algebra_transport, dictionary_update, ridge_solve
from kam.phase6.manifest import build_stage_manifest, build_stage_rows, load_config
from kam.phase6.gates import evaluate_stage_results
from kam.phase6.overnight_manifest import (
    build_preflight_rows,
    build_wave1_rows,
    build_wave2_rows,
    build_wave3_rows,
    write_manifest,
)
from kam.phase6.overnight_runner import _optimization_groups, run_row as run_overnight_row
from kam.phase6.run_array import run_row
from kam.phase6.stats import bootstrap_ci, equivalence_test, holm_adjust, paired_effect
from kam.transformer import build_baseline


def test_all_named_phase6_architectures_forward() -> None:
    for name in ("T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L", "T-KAM-ALT", "T-KAM-VP", "T-KAM-ONLINE", "T-KAM-DUAL"):
        model, spec = build_baseline(name, scale="tiny", vocab_size=19, max_seq_len=8, num_supports=8, top_k=2)
        logits = model(torch.randint(19, (1, 8)))
        assert logits.shape == (1, 8, 19)
        assert spec.name == name


def test_phase6_transformer_budget_matching() -> None:
    target = 2_000_000
    for name in ("T0", "T-WIDE", "T-MEMTOK", "T-MOE", "T-PKM", "T-KAM-F", "T-KAM-L"):
        model, _ = build_baseline(name, scale="2M", vocab_size=19, max_seq_len=8, num_supports=32, top_k=2, target_parameters=target)
        total = sum(parameter.numel() for parameter in model.parameters())
        assert abs(total - target) / target < 0.02
        if name == "T-WIDE":
            assert model.config.feedforward_dim > 4 * model.config.d_model
    kam_model, _ = build_baseline("T-KAM-L", scale="2M", vocab_size=19, max_seq_len=8, num_supports=32, top_k=2, target_parameters=target)
    assert kam_model.active_parameters_per_token < sum(parameter.numel() for parameter in kam_model.parameters())
    calibrated, _ = build_baseline("T0", scale="2M", vocab_size=19, max_seq_len=8, target_parameters=1_000_000)
    calibrated_total = sum(parameter.numel() for parameter in calibrated.parameters())
    assert abs(calibrated_total - 1_000_000) / 1_000_000 < 0.03


def test_phase6_overnight_manifests_have_fixed_graph_sizes(tmp_path) -> None:
    groups = (
        (build_preflight_rows(), 4, "preflight"),
        (build_wave1_rows(), 32, "wave1"),
        (build_wave2_rows([]), 16, "wave2"),
        (build_wave3_rows([]), 8, "wave3"),
    )
    for rows, expected, wave in groups:
        assert len(rows) == expected
        assert len({row["row_id"] for row in rows}) == expected
        assert {row["wave"] for row in rows} == {wave}
        metadata = write_manifest(rows, tmp_path / f"{wave}.jsonl")
        assert metadata["rows"] == expected
        assert len(metadata["sha256"]) == 64
    assert sum(row["target_seconds"] for rows, _, _ in groups for row in rows) / 3600 == 45.733333333333334


def test_phase6_learned_geometry_is_optimized_before_final_freeze() -> None:
    joint, _ = build_baseline("T-KAM-L", scale="tiny", vocab_size=19, max_seq_len=8, num_supports=8, top_k=2)
    algebra, geometry = _optimization_groups(joint, "joint_sgd")
    assert geometry
    assert all(any(parameter is candidate for candidate in algebra) for parameter in geometry)
    fixed, _ = build_baseline("T-KAM-F", scale="tiny", vocab_size=19, max_seq_len=8, num_supports=8, top_k=2)
    _, fixed_geometry = _optimization_groups(fixed, "joint_sgd")
    assert not fixed_geometry
    vp, _ = build_baseline("T-KAM-VP", scale="tiny", vocab_size=19, max_seq_len=8, num_supports=8, top_k=2)
    _, vp_geometry = _optimization_groups(vp, "vp_stop_gradient")
    assert not vp_geometry


def test_phase6_overnight_language_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PHASE6_OVERNIGHT_SMOKE_SECONDS", "0.05")
    row = dict(build_preflight_rows()[0])
    row.update(scale="tiny", target_parameter_budget=100_000, batch_size=1, sequence_length=8, num_supports=8, minimum_tokens=8)
    result = run_overnight_row(row, device="cpu", output_root=tmp_path)
    assert result["status"] == "pass"
    assert result["metrics"]["smoke_override"]
    assert result["metrics"]["tokens"] >= 8


def test_phase6_approximate_router_and_initializers() -> None:
    torch.manual_seed(1)
    data = torch.randn(20, 6)
    assert fixed_data_sample(data, 5, seed=1).shape == (5, 6)
    assert kmeans(data, 5, seed=1).shape == (5, 6)
    router = ApproximateTopKRouter(top_k=2, candidate_size=8, seed=1)
    route = router(data[:3], data)
    assert route.indices.shape == (3, 2)


def test_phase6_algebra_transport_and_dictionary_update() -> None:
    torch.manual_seed(2)
    x = torch.randn(24, 4)
    beta = torch.randn(4)
    old = ridge_solve(x, x @ beta).solution
    transported = algebra_transport(x, x + 0.01, old)
    assert float(transported["transport_error"]) < 0.1
    updated, diagnostics = dictionary_update(x, x[:6])
    assert updated.shape == (6, 4)
    assert 0 <= diagnostics["coverage"] <= 1
    decision = GeometryTrustRegion().evaluate(x, x, x @ beta.unsqueeze(-1), x @ beta.unsqueeze(-1), objective_old=1, objective_new=0.9, support_utilization=1, condition_number=1)
    assert decision.accepted


def test_phase6_task_lanes_and_statistics() -> None:
    inputs, targets = variable_copy(batch=3, payload_length=4)
    assert inputs.shape == targets.shape
    assert mqar(batch=3, pairs=2, sequence_length=12)[0].shape == (3, 12)
    symbols, next_symbols, regimes = controlled_symbolic_regimes(batch=3, length=8)
    assert symbols.shape == next_symbols.shape and regimes.shape == (3,)
    tokens, _ = load_text_tokens()
    assert language_batches(tokens, batch_size=2, sequence_length=8)[0].shape == (2, 8)
    effect = paired_effect([1, 2, 3], [2, 3, 4])
    assert effect["mean_difference"] == 1
    assert bootstrap_ci([1, 2, 3])[0] <= 2 <= bootstrap_ci([1, 2, 3])[1]
    assert equivalence_test([1, 2, 3], [1.01, 2.01, 3.01], margin=0.1)["equivalent"]
    assert holm_adjust({"a": 0.01, "b": 0.04})["a"] <= holm_adjust({"a": 0.01, "b": 0.04})["b"]


def test_phase6_profile_manifests_are_static_and_unique(tmp_path) -> None:
    for stage in ("stage1_mechanism", "stage2_transformer_comparison", "stage3_router_scaling", "stage4_online_adaptation", "stage5_long_training", "stage6_confirmation"):
        config_path = f"configs/phase6/{stage}.yaml"
        rows = build_stage_manifest(config_path, tmp_path / f"{stage}.jsonl", mode="profile")
        assert rows
        assert len({row["row_id"] for row in rows}) == len(rows)
        assert all(row["stage"] == stage for row in rows)
        if stage == "stage1_mechanism":
            assert {row["geometry"] for row in rows} >= {
                "fixed_random",
                "fixed_data_sample",
                "fixed_kmeans",
                "fixed_farthest_point",
                "learned_full",
                "learned_low_rank_delta",
            }
        if stage == "stage2_transformer_comparison":
            assert {row["scale"] for row in rows} <= {"2M", "10M", "30M"}
            assert {row["target_parameter_budget"] for row in rows} <= {2_000_000, 10_000_000, 30_000_000}
        if stage == "stage5_long_training":
            assert {row["scale"] for row in rows} <= {"10M", "30M"}
            assert {row["training_token_cap"] for row in rows} == {4096}
        if stage == "stage6_confirmation":
            assert {row["scale"] for row in rows} == {"10M"}
            assert {row["target_parameter_budget"] for row in rows} == {10_000_000}


def test_phase6_full_manifest_does_not_overwrite_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "stage5_long_training.yaml"
    config.write_text("stage: stage5_long_training\nseed: 6505\n", encoding="utf-8")
    profile = build_stage_manifest(config, mode="profile")
    full = build_stage_manifest(config, mode="full")
    profile_path = tmp_path / "results/phase6/stage5_long_training/manifests/profile.jsonl"
    full_path = tmp_path / "results/phase6/stage5_long_training/manifests/full.jsonl"
    profile_summary = tmp_path / "results/phase6/stage5_long_training/manifests/profile_summary.json"
    full_summary = tmp_path / "results/phase6/stage5_long_training/manifests/full_summary.json"
    assert len(profile) == 12
    assert len(full) == 100
    assert profile_path.exists() and full_path.exists()
    assert profile_summary.exists() and full_summary.exists()
    assert profile_path.read_text(encoding="utf-8").count("\n") == 12
    assert full_path.read_text(encoding="utf-8").count("\n") == 100


def test_phase6_generic_gate_reports_incomplete_arrays() -> None:
    result = evaluate_stage_results(
        [{"row_id": "row_0", "status": "pass", "metrics": {"loss": 1.0}}],
        expected=2,
    )
    assert not result["stage_pass"]
    assert not result["row_count_matches_expected"]
    assert result["missing_row_count"] == 1
    assert result["extra_row_count"] == 0


def test_phase6_generic_gate_rejects_nonfinite_metrics() -> None:
    result = evaluate_stage_results(
        [{"row_id": "row_inf", "status": "pass", "metrics": {"loss": float("inf"), "loss_history": [1.0, float("nan")]}}],
        expected=1,
    )
    assert not result["stage_pass"]
    assert result["nonfinite_row_ids"] == ["row_inf"]


def test_phase6_router_scaling_honors_precision_and_mode() -> None:
    for precision in ("fp32", "bf16", "fp16"):
        profile = run_row(
            {
                "stage": "stage3_router_scaling",
                "stage_mode": "profile",
                "row_id": f"router_{precision}",
                "seed": 7,
                "router": "exact",
                "slot": 64,
                "top_k": 4,
                "precision": precision,
            },
            device="cpu",
        )
        assert profile["status"] == "pass"
        assert profile["metrics"]["requested_supports"] == 64
        assert profile["metrics"]["benchmark_supports"] == 64
        assert profile["metrics"]["precision_requested"] == precision
        assert profile["metrics"]["precision_effective"] == "fp32"
    full = run_row(
        {
            "stage": "stage3_router_scaling",
            "stage_mode": "full",
            "row_id": "router_full",
            "seed": 7,
            "router": "chunked",
            "slot": 128,
            "top_k": 4,
            "precision": "fp32",
        },
        device="cpu",
    )
    assert full["status"] == "pass"
    assert full["metrics"]["benchmark_supports"] == 128
    product = run_row(
        {
            "stage": "stage3_router_scaling",
            "stage_mode": "profile",
            "row_id": "router_product_key",
            "seed": 7,
            "router": "product_key",
            "slot": 1000,
            "top_k": 4,
            "precision": "fp32",
        },
        device="cpu",
    )
    assert product["status"] == "pass"
    assert product["metrics"]["benchmark_supports"] == 1024
    assert product["metrics"]["recall_at_k_against_exact"] == 1.0


def test_phase6_runner_respects_task_and_optimizer_factors() -> None:
    mechanism_config = load_config("configs/phase6/stage1_mechanism.yaml")
    mechanism_row = dict(next(row for row in build_stage_rows(mechanism_config, mode="profile") if row["architecture"] == "T-KAM-L"))
    mechanism_row.update(task="mqar", optimizer="ridge_resolve", row_id="runner_factor_smoke")
    mechanism_result = run_row(mechanism_row, device="cpu")
    assert mechanism_result["status"] == "pass"
    assert mechanism_result["metrics"]["task_mask_fraction"] < 1
    assert mechanism_result["metrics"]["optimizer_mode"] == "ridge_resolve"

    wide_base = dict(next(row for row in build_stage_rows(mechanism_config, mode="profile") if row["architecture"] == "T-WIDE"))
    for optimizer in ("ridge_resolve", "variable_projection_implicit"):
        wide_row = dict(wide_base)
        wide_row.update(task="mqar", optimizer=optimizer, row_id=f"runner_wide_{optimizer}")
        wide_result = run_row(wide_row, device="cpu")
        assert wide_result["status"] == "pass"
        assert wide_result["metrics"]["optimizer_mode"] == optimizer

    alternating_row = dict(mechanism_row)
    alternating_row.update(
        geometry="learned_full",
        optimizer="alternating_8_1",
        fidelity=0.05,
        row_id="runner_alternating_geometry",
    )
    alternating_result = run_row(alternating_row, device="cpu")
    assert alternating_result["status"] == "pass"
    assert alternating_result["metrics"]["alternating_geometry_steps"] >= 1

    transformer_config = load_config("configs/phase6/stage2_transformer_comparison.yaml")
    transformer_rows = build_stage_rows(transformer_config, mode="profile")
    for task in ("mqar", "controlled_symbolic_regimes", "small_language", "prototype"):
        row = dict(next(candidate for candidate in transformer_rows if candidate["task"] == task))
        row.update(scale="tiny", row_id=f"runner_{task}")
        result = run_row(row, device="cpu")
        assert result["status"] == "pass"
        assert result["metrics"]["task"] == task
    dynamics_row = {
        "stage": "stage5_long_training",
        "stage_mode": "profile",
        "task": "switching_mackey_glass",
        "architecture": "T0",
        "scale": "tiny",
        "seed": 17,
        "row_id": "runner_switching_mackey_glass",
        "training_steps": 1,
        "num_supports": 32,
        "top_k": 4,
    }
    dynamics_result = run_row(dynamics_row, device="cpu")
    assert dynamics_result["status"] == "pass"
    assert dynamics_result["metrics"]["task"] == "switching_mackey_glass"
    alt_row = dict(next(candidate for candidate in transformer_rows if candidate["task"] == "prototype"))
    alt_row.update(scale="tiny", architecture="T-KAM-ALT", row_id="runner_transformer_alt")
    alt_result = run_row(alt_row, device="cpu")
    assert alt_result["status"] == "pass"
    assert alt_result["metrics"]["training_optimizer_mode"] == "alternating_8_1"
    assert alt_result["metrics"]["geometry_update_steps"] >= 1
    vp_row = dict(alt_row)
    vp_row.update(architecture="T-KAM-VP", row_id="runner_transformer_vp")
    vp_result = run_row(vp_row, device="cpu")
    assert vp_result["status"] == "pass"
    assert vp_result["metrics"]["training_optimizer_mode"] == "variable_projection_stopgrad"
    assert vp_result["metrics"]["geometry_update_steps"] == 0

    online_config = load_config("configs/phase6/stage4_online_adaptation.yaml")
    online_row = next(row for row in build_stage_rows(online_config, mode="profile") if row["task"] == "symbolic_schedule")
    online_result = run_row(online_row, device="cpu")
    assert online_result["status"] == "pass"
    assert online_result["metrics"]["schedule_segments"] == 5
    online_memory_row = dict(online_row)
    online_memory_row.update(architecture="T-KAM-ONLINE", adapter="episodic_insertion", row_id="runner_online_memory")
    online_memory_result = run_row(online_memory_row, device="cpu")
    assert online_memory_result["status"] == "pass"
    assert online_memory_result["metrics"]["memory_used"] == 1
    assert online_memory_result["metrics"]["episodic_active"] == 1
