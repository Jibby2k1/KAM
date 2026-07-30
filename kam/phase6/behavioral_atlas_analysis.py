"""Audits, forecasts, reports, and Stage 0 figures for the behavioral atlas."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kam.phase6.behavioral_atlas_manifest import CAMPAIGN
from kam.phase6.behavioral_atlas_runner import build_behavioral_atlas_model


COLORS = {
    "T0": "#475569",
    "fixed_keys": "#64748b",
    "learned_joint_adamw": "#a21caf",
    "learned_joint_freeze80": "#0f766e",
    "learned_alt8_freeze80": "#1d4ed8",
    "learned_alt32_freeze80": "#c2410c",
}

LINESTYLES = {
    "T0": "-",
    "fixed_keys": "--",
    "learned_joint_adamw": "-.",
    "learned_joint_freeze80": ":",
    "learned_alt8_freeze80": (0, (5, 2)),
    "learned_alt32_freeze80": (0, (3, 1, 1, 1)),
}


def load_results(run_root: str | Path) -> list[dict[str, Any]]:
    root = Path(run_root) / "rows" / "behavioral_atlas_v2"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _freeze_integrity(row: dict[str, Any]) -> bool:
    has_geometry = str(row.get("architecture")) != "T0"
    freeze_fraction = float(row.get("freeze_fraction", 1.0))
    fixed = not bool(row.get("geometry_trainable", True))
    if not has_geometry or (not fixed and freeze_fraction >= 1.0):
        return not bool(row.get("postfreeze_key_grad_observed"))
    return bool(row.get("postfreeze_key_hash_unchanged")) and float(row.get("postfreeze_relative_l2_drift", math.inf)) <= 1e-12 and not bool(row.get("postfreeze_key_grad_observed"))


def audit_results(results: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {str(row["row_id"]) for row in manifest_rows}
    observed = [str(row.get("row_id")) for row in results]
    identity: dict[tuple[int, str], set[str]] = defaultdict(set)
    anchors: dict[tuple[int, int], set[str]] = defaultdict(set)
    sample_orders: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in results:
        identity[(int(row["seed"]), str(row.get("identity_group")))].add(str(row.get("initial_state_hash")))
        anchor_size = int(row.get("anchor_token_states_resolved", 0))
        anchors[(int(row["seed"]), anchor_size)].add(str(row.get("anchor_sha256")))
        sample_orders[(int(row["seed"]), int(row.get("target_tokens_resolved", 0)))].add(str(row.get("sample_order_sha256")))
    trace_complete = True
    for row in results:
        if str(row.get("trace_level")) == "off":
            continue
        for point in row.get("traces", []):
            behavior = point.get("behavior")
            trace_complete = trace_complete and isinstance(behavior, dict) and behavior.get("anchor_sha256") == row.get("anchor_sha256")
    checks = {
        "all_rows_present": set(observed) == expected and len(observed) == len(expected),
        "all_rows_passed": len(results) == len(manifest_rows) and all(row.get("status") == "pass" for row in results),
        "initial_states_identical_within_seed_and_identity_group": bool(identity) and all(len(values) == 1 for values in identity.values()),
        "anchors_identical_within_seed_and_size": bool(anchors) and all(len(values) == 1 for values in anchors.values()),
        "sample_order_identical_within_seed_and_budget": bool(sample_orders) and all(len(values) == 1 for values in sample_orders.values()),
        "finite_metrics": bool(results) and all(_all_finite(row.get("traces")) and _all_finite(row.get("test_loss")) for row in results),
        "executable_optimizer_provenance": bool(results) and all(
            row.get("optimizer_provenance", {}).get("effective_optimizer_class") == "AdamW"
            and row.get("optimizer_provenance", {}).get("label_matches_executable")
            for row in results
        ),
        "standard_trace_complete": trace_complete,
        "freeze_integrity": all(_freeze_integrity(row) for row in results),
        "permutation_symmetry": all(row.get("matched_key_expert_permutation", {}).get("passed") for row in results),
        "permutation_operational_stability": all(
            row.get("matched_key_expert_permutation", {}).get("operational_within_expected_precision_tolerance", True)
            for row in results
        ),
        "restart_identity_when_registered": all(row.get("restart_state_hash_match") is True for row in results if row.get("save_snapshots")),
    }
    stage = str(manifest_rows[0].get("stage")) if manifest_rows else "unknown"
    if stage.startswith("l4_profile"):
        checks.update({
            "actual_nvidia_l4": bool(results) and all("L4" in str(row.get("execution", {}).get("device_name")) for row in results),
            "bf16_tf32_fused_adamw": bool(results) and all(
                row.get("execution", {}).get("bf16_supported")
                and row.get("execution", {}).get("tf32")
                and row.get("execution", {}).get("fused_adamw")
                for row in results
            ),
            "bounded_profile_budget": bool(results) and all(int(row.get("target_tokens_resolved", 0)) <= 250_000 for row in results),
        })
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_rows": len(manifest_rows),
        "observed_rows": len(results),
        "missing_row_ids": sorted(expected - set(observed)),
    }


def _final_behavior(row: dict[str, Any]) -> dict[str, Any]:
    for point in reversed(row.get("traces", [])):
        if isinstance(point.get("behavior"), dict):
            return point["behavior"]
    return {}


def profile_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = {str(row.get("profile_kind")): row for row in results if row.get("profile_kind")}
    output: dict[str, Any] = {}
    off = profiles.get("trace_off")
    standard = profiles.get("standard_trace")
    if off and standard:
        output["trace_overhead_fraction"] = float(standard["wall_seconds"] / max(float(off["wall_seconds"]), 1e-30) - 1)
        output["trace_overhead_pass"] = output["trace_overhead_fraction"] <= 0.10
    standard_behavior = _final_behavior(standard) if standard else {}
    doubled_behavior = _final_behavior(profiles.get("doubled_anchor", {})) if profiles.get("doubled_anchor") else {}
    if standard_behavior and doubled_behavior:
        fields = (
            "memory_output_stable_rank_mean",
            "memory_output_participation_ratio_mean",
            "memory_contribution_ratio_mean",
        )
        differences = {
            field: abs(float(standard_behavior[field]) - float(doubled_behavior[field])) / max(abs(float(doubled_behavior[field])), 1e-12)
            for field in fields
        }
        for field in ("normalized_support_entropy", "dead_support_fraction"):
            standard_value = standard_behavior["routing_decomposition"]["states"]["Qt_Kt"][field]
            doubled_value = doubled_behavior["routing_decomposition"]["states"]["Qt_Kt"][field]
            differences[field] = abs(float(standard_value) - float(doubled_value)) / max(abs(float(doubled_value)), 1e-12)
        output["anchor_relative_differences"] = differences
        output["anchor_sufficiency_pass"] = max(differences.values()) <= 0.05
    repeat_a = profiles.get("repeatability_a")
    repeat_b = profiles.get("repeatability_b")
    if repeat_a and repeat_b:
        output["repeatability"] = {
            "final_key_hash_equal": repeat_a.get("final_key_hash") == repeat_b.get("final_key_hash"),
            "validation_abs_difference": abs(float(repeat_a["validation_loss"]) - float(repeat_b["validation_loss"])),
            "test_abs_difference": abs(float(repeat_a["test_loss"]) - float(repeat_b["test_loss"])),
        }
        output["repeatability_pass"] = output["repeatability"]["final_key_hash_equal"] and output["repeatability"]["test_abs_difference"] <= 1e-5
    compiled = profiles.get("compile_candidate")
    if standard and compiled:
        output["compile"] = {
            "applied": bool(compiled.get("execution", {}).get("compile_applied")),
            "speedup_including_compile_cost": float(compiled["tokens_per_second"]) / max(float(standard["tokens_per_second"]), 1e-30),
            "eligible_for_later_use": bool(compiled.get("execution", {}).get("compile_applied")) and float(compiled["tokens_per_second"]) >= 1.10 * float(standard["tokens_per_second"]),
        }
    return output


def forecast_manifest(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_bytes = 0
    row_forecasts = []
    for row in manifest_rows:
        model = build_behavioral_atlas_model(row)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        keys = sum(layer.keys.numel() for layer in model.memory_layers)
        checkpoints = len(set(int(value) for value in row.get("validation_token_checkpoints", [])))
        full_snapshots = 2 if row.get("save_snapshots") else 0
        model_bytes = full_snapshots * parameters * 4
        key_bytes = checkpoints * keys * 2 if row.get("save_snapshots") else 0
        trace_bytes = checkpoints * 300_000
        row_bytes = model_bytes + key_bytes + trace_bytes
        total_bytes += row_bytes
        row_forecasts.append({"row_id": row["row_id"], "parameter_count": parameters, "forecast_bytes": row_bytes})
    target_tokens = sum(int(row["target_tokens"]) for row in manifest_rows)
    estimated_gpu_hours = target_tokens / 50_000_000 * 4.0 * 1.10
    return {
        "rows": len(manifest_rows),
        "forecast_bytes": int(total_bytes),
        "forecast_gib": total_bytes / 2**30,
        "required_with_25_percent_headroom_gib": total_bytes * 1.25 / 2**30,
        "estimated_l4_gpu_hours": estimated_gpu_hours,
        "estimated_four_l4_wall_hours": estimated_gpu_hours / 4,
        "row_forecasts": row_forecasts,
    }


def _save(figure, root: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    return paths


def build_figures(results: list[dict[str, Any]], report_root: str | Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(report_root) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    functional_results = [row for row in results if row.get("profile_kind") is None] or results

    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [str(row.get("profile_kind") or row["arm"]) for row in results]
    values = [float(row["tokens_per_second"]) for row in results]
    axis.barh(range(len(labels)), values, color=[COLORS.get(str(row["arm"]), "#64748b") for row in results])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Training tokens per second")
    axis.set_title("Stage 0 L4/runner throughput")
    axis.grid(axis="x", alpha=.2)
    paths += _save(figure, root, "stage0_runtime_throughput")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    states = ("Q0_K0", "Qt_K0", "Q0_Kt", "Qt_Kt")
    plotted = False
    for row in functional_results:
        behavior = _final_behavior(row)
        routing = behavior.get("routing_decomposition", {}).get("states", {})
        if not routing:
            continue
        axis.plot(range(4), [routing[state]["jaccard_to_Q0_K0"] for state in states], marker="o", color=COLORS.get(str(row["arm"]), "#64748b"), linestyle=LINESTYLES.get(str(row["arm"]), "-"), alpha=.7, label=str(row["arm"]))
        plotted = True
    axis.set_xticks(range(4), states)
    axis.set_ylabel("Top-k Jaccard to initial routing")
    axis.set_ylim(0, 1.02)
    axis.set_title("Query-versus-key routing decomposition")
    axis.grid(axis="y", alpha=.2)
    if plotted:
        handles, names = axis.get_legend_handles_labels(); unique = dict(zip(names, handles)); axis.legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_routing_decomposition")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in functional_results:
        behavior = _final_behavior(row)
        if not behavior:
            continue
        axes[0].scatter(behavior.get("memory_contribution_ratio_mean", 0), behavior.get("memory_output_stable_rank_mean", 0), color=COLORS.get(str(row["arm"]), "#64748b"), label=str(row["arm"]), alpha=.75)
        axes[1].scatter(behavior.get("anchor_logit_l2_drift", 0), float(row["validation_loss"]), color=COLORS.get(str(row["arm"]), "#64748b"), alpha=.75)
    axes[0].set_xlabel("Memory contribution ratio"); axes[0].set_ylabel("Memory-output stable rank"); axes[0].set_title("Realized branch capacity")
    axes[1].set_xlabel("Anchor-logit drift from initialization"); axes[1].set_ylabel("Final validation loss"); axes[1].set_title("Functional motion and fit")
    for axis in axes: axis.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels(); unique = dict(zip(names, handles)); axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_functional_behavior")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for row in functional_results:
        points = [(int(point["tokens"]), float(point["groups"]["memory_keys"]["cumulative_relative_l2_delta_from_initial"])) for point in row.get("traces", []) if point.get("phase") != "freeze_event"]
        if not points or str(row.get("architecture")) == "T0":
            continue
        axis.plot([token / 1e6 for token, _ in points], [value for _, value in points], color=COLORS.get(str(row["arm"]), "#64748b"), linestyle=LINESTYLES.get(str(row["arm"]), "-"), label=str(row["arm"]), alpha=.7)
    axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel("Relative key drift"); axis.set_title("Stage 0 memory-key displacement"); axis.grid(alpha=.2)
    handles, names = axis.get_legend_handles_labels(); unique = dict(zip(names, handles)); axis.legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_key_drift")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 6))
    learning: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    learning_seeds: dict[str, set[int]] = defaultdict(set)
    for row in functional_results:
        arm = str(row["arm"]); learning_seeds[arm].add(int(row["seed"]))
        for point in row.get("traces", []):
            if point.get("phase") != "freeze_event":
                learning[arm][int(point["tokens"])].append(float(point["validation_loss"]))
    for arm, by_token in learning.items():
        tokens = sorted(by_token)
        center = [float(np.median(by_token[token])) for token in tokens]
        low = [float(np.min(by_token[token])) for token in tokens]
        high = [float(np.max(by_token[token])) for token in tokens]
        color = COLORS.get(arm, "#64748b")
        axis.fill_between([token / 1e6 for token in tokens], low, high, color=color, alpha=.12)
        axis.plot([token / 1e6 for token in tokens], center, marker="o", color=color, linestyle=LINESTYLES.get(arm, "-"), label=f"{arm} (n={len(learning_seeds[arm])})")
    axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel("Validation cross-entropy loss"); axis.set_title("Stage 0 validation learning curves (median and seed range)"); axis.grid(alpha=.2)
    handles, names = axis.get_legend_handles_labels(); axis.legend(handles, names, fontsize=7)
    paths += _save(figure, root, "stage0_learning_curves")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in functional_results:
        color = COLORS.get(str(row["arm"]), "#64748b")
        points = [(int(point["tokens"]), point.get("window_dynamics", {}).get("memory_keys", {})) for point in row.get("traces", []) if point.get("phase") != "freeze_event"]
        gradient = [(token, metrics.get("raw_gradient_l2_norm", {}).get("median")) for token, metrics in points]
        update = [(token, metrics.get("update_to_weight_ratio", {}).get("median")) for token, metrics in points]
        gradient = [(token, value) for token, value in gradient if value is not None]
        update = [(token, value) for token, value in update if value is not None]
        if gradient:
            axes[0].plot([token / 1e6 for token, _ in gradient], [value for _, value in gradient], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), label=str(row.get("profile_kind") or row["arm"]), alpha=.7)
        if update:
            axes[1].plot([token / 1e6 for token, _ in update], [value for _, value in update], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), label=str(row.get("profile_kind") or row["arm"]), alpha=.7)
    axes[0].set_ylabel("Median raw key-gradient L2 norm"); axes[1].set_ylabel("Median key update / weight norm")
    for axis in axes:
        axis.set_xlabel("Training tokens (millions)"); axis.set_yscale("symlog", linthresh=1e-10); axis.grid(alpha=.2)
    axes[0].set_title("Memory-key gradient trajectory"); axes[1].set_title("Memory-key optimizer-update trajectory")
    handles, names = axes[0].get_legend_handles_labels(); unique = dict(zip(names, handles)); axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_key_gradient_update_dynamics")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in functional_results:
        color = COLORS.get(str(row["arm"]), "#64748b")
        points = [(int(point["tokens"]), point.get("behavior")) for point in row.get("traces", []) if isinstance(point.get("behavior"), dict)]
        kl = [(token, max(float(behavior.get("anchor_predictive_kl_to_initial", 0)), 1e-12)) for token, behavior in points]
        flips = [(token, float(behavior.get("anchor_top1_flip_rate", 0))) for token, behavior in points]
        if kl:
            axes[0].plot([token / 1e6 for token, _ in kl], [value for _, value in kl], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), label=str(row.get("profile_kind") or row["arm"]), alpha=.7)
            axes[1].plot([token / 1e6 for token, _ in flips], [value for _, value in flips], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), alpha=.7)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Predictive KL to initialization (floor 1e-12)"); axes[1].set_ylabel("Anchor top-1 change rate")
    for axis in axes:
        axis.set_xlabel("Training tokens (millions)"); axis.grid(alpha=.2)
    axes[0].set_title("Anchor prediction-distribution drift"); axes[1].set_title("Anchor top-1 prediction drift")
    handles, names = axes[0].get_legend_handles_labels(); unique = dict(zip(names, handles)); axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_anchor_prediction_drift")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for row in functional_results:
        color = COLORS.get(str(row["arm"]), "#64748b")
        points = [(int(point["tokens"]), point.get("behavior")) for point in row.get("traces", []) if isinstance(point.get("behavior"), dict)]
        routing = [(token, behavior.get("routing_decomposition", {}).get("states", {}).get("Qt_Kt", {}), behavior) for token, behavior in points]
        routing = [(token, state, behavior) for token, state, behavior in routing if state]
        if routing:
            label = str(row.get("profile_kind") or row["arm"])
            x = [token / 1e6 for token, _, _ in routing]
            axes[0].plot(x, [state.get("normalized_support_entropy", 0) for _, state, _ in routing], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), label=label, alpha=.7)
            axes[1].plot(x, [state.get("dead_support_fraction", 0) for _, state, _ in routing], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), alpha=.7)
            axes[2].plot(x, [behavior.get("memory_contribution_ratio_mean", 0) for _, _, behavior in routing], marker="o", color=color, linestyle=LINESTYLES.get(str(row["arm"]), "-"), alpha=.7)
    axes[0].set_ylabel("Normalized support entropy"); axes[1].set_ylabel("Dead-support fraction"); axes[2].set_ylabel("Memory contribution ratio")
    for axis in axes:
        axis.set_xlabel("Training tokens (millions)"); axis.set_ylim(bottom=0); axis.grid(alpha=.2)
    axes[0].set_title("Support-use diversity"); axes[1].set_title("Unused memory supports"); axes[2].set_title("Memory branch contribution")
    handles, names = axes[0].get_legend_handles_labels(); unique = dict(zip(names, handles)); axes[0].legend(unique.values(), unique.keys(), fontsize=7)
    paths += _save(figure, root, "stage0_support_and_contribution_dynamics")
    plt.close(figure)
    return paths


def descriptive_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose compact row-grain science and systems metrics in reports."""
    output: list[dict[str, Any]] = []
    for row in results:
        traces = [point for point in row.get("traces", []) if point.get("phase") != "freeze_event"]
        initial = traces[0] if traces else {}
        final = traces[-1] if traces else {}
        behavior = _final_behavior(row)
        routing = behavior.get("routing_decomposition", {}).get("states", {}).get("Qt_Kt", {})
        key_group = final.get("groups", {}).get("memory_keys", {})
        symmetry = row.get("matched_key_expert_permutation", {})
        output.append({
            "arm": row.get("arm"),
            "seed": row.get("seed"),
            "profile_kind": row.get("profile_kind"),
            "initial_validation_loss": initial.get("validation_loss"),
            "final_validation_loss": final.get("validation_loss"),
            "test_loss": row.get("test_loss"),
            "relative_key_drift": key_group.get("cumulative_relative_l2_delta_from_initial"),
            "postfreeze_relative_key_drift": row.get("postfreeze_relative_l2_drift"),
            "routing_jaccard_to_initial": routing.get("jaccard_to_Q0_K0"),
            "support_entropy": routing.get("normalized_support_entropy"),
            "dead_support_fraction": routing.get("dead_support_fraction"),
            "effective_support_count": routing.get("global_effective_support_count"),
            "memory_contribution_ratio": behavior.get("memory_contribution_ratio_mean"),
            "memory_output_stable_rank": behavior.get("memory_output_stable_rank_mean"),
            "tokens_per_second": row.get("tokens_per_second"),
            "peak_vram_mib": float(row.get("peak_vram_bytes", 0)) / 1048576,
            "semantic_permutation_max_abs_logit_difference": symmetry.get("max_abs_logit_difference"),
            "operational_permutation_max_abs_logit_difference": symmetry.get("operational_max_abs_logit_difference"),
            "operational_permutation_top1_flip_rate": symmetry.get("operational_top1_flip_rate"),
            "operational_permutation_predictive_kl": symmetry.get("operational_predictive_kl"),
        })
    return output


def analyze_behavioral_atlas(run_root: str | Path, report_root: str | Path, manifest: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = load_results(run_root)
    audit = audit_results(results, manifest_rows)
    profiles = profile_metrics(results)
    descriptive = descriptive_rows(results)
    forecast = forecast_manifest(manifest_rows)
    figures = build_figures(results, report_root) if results else []
    stage = str(manifest_rows[0].get("stage")) if manifest_rows else "unknown"
    if stage.startswith("l4_profile"):
        decision = "L4_PROFILE_PASS" if audit["passed"] else "L4_PROFILE_BLOCKED"
    else:
        required_profile_checks = [value for key, value in profiles.items() if key.endswith("_pass")]
        decision = "STAGE0_PASS" if audit["passed"] and required_profile_checks and all(required_profile_checks) else "STAGE0_BLOCKED"
    summary = {
        "campaign": CAMPAIGN,
        "stage": stage,
        "decision": decision,
        "rows": len(results),
        "audit": audit,
        "profile_metrics": profiles,
        "descriptive_rows": descriptive,
        "forecast": forecast,
        "figures": figures,
    }
    (run_root / "behavioral_atlas_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "behavioral_atlas_forecast.json").write_text(json.dumps(forecast, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 6.2 Stage 0 Behavioral Atlas Report",
        "",
        f"**Stage:** `{stage}`",
        f"**Decision:** `{decision}`",
        "",
        f"- Complete rows: {len(results)}/{len(manifest_rows)}",
        f"- Audit passed: `{audit['passed']}`",
        f"- Figures: {len(figures) // 2} PNG/SVG pairs",
        f"- Forecast storage with headroom: {forecast['required_with_25_percent_headroom_gib']:.2f} GiB",
        f"- Forecast L4 GPU-hours: {forecast['estimated_l4_gpu_hours']:.2f}",
        "",
        "## Audit",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in audit["checks"].items())
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines.extend([
        "",
        "## Descriptive results",
        "",
        "| Arm | Seed | Profile | Validation loss (initial → final) | Test loss | Key drift | Post-freeze drift | Route Jaccard | Support entropy | Dead supports | Memory contribution | Stable rank | tokens/s | VRAM MiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in descriptive:
        validation = f"{fmt(item["initial_validation_loss"])} → {fmt(item["final_validation_loss"])}"
        lines.append(
            f"| {item["arm"]} | {item["seed"]} | {item["profile_kind"] or "—"} | {validation} | "
            f"{fmt(item["test_loss"])} | {fmt(item["relative_key_drift"])} | "
            f"{fmt(item["postfreeze_relative_key_drift"])} | {fmt(item["routing_jaccard_to_initial"])} | "
            f"{fmt(item["support_entropy"])} | {fmt(item["dead_support_fraction"])} | "
            f"{fmt(item["memory_contribution_ratio"])} | {fmt(item["memory_output_stable_rank"])} | "
            f"{fmt(item["tokens_per_second"])} | {fmt(item["peak_vram_mib"])} |"
        )
    if descriptive:
        lines.extend(["", "### Permutation precision diagnostics", "", "```json", json.dumps([{key: value for key, value in item.items() if "permutation" in key} for item in descriptive], indent=2, sort_keys=True), "```"])
    if profiles:
        lines.extend(["", "## Profile comparison gates", "", "```json", json.dumps(profiles, indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Interpretation boundary", "", "Stage 0 validates measurement, repeatability, runtime, storage, and execution semantics. It is excluded from scientific inference and cannot promote an architecture."])
    (report_root / "BEHAVIORAL_ATLAS_STAGE0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


__all__ = ["analyze_behavioral_atlas", "audit_results", "build_figures", "forecast_manifest", "load_results", "profile_metrics"]
