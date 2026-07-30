"""Inferential analysis and reports for Phase 6.2 Stage 1."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kam.phase6.behavioral_atlas_analysis import audit_results, load_results

PRIMARY_COMPARISONS = (
    ("learned_joint_adamw_no_freeze", "fixed_keys"),
    ("learned_joint_adamw_freeze80", "fixed_keys"),
    ("learned_joint_adamw_freeze80", "learned_joint_adamw_no_freeze"),
    ("learned_alt8_adamw_freeze80", "learned_joint_adamw_freeze80"),
)
SECONDARY_COMPARISONS = (
    ("learned_joint_adamw_freeze25", "learned_joint_adamw_freeze80"),
    ("learned_joint_adamw_freeze50", "learned_joint_adamw_freeze80"),
    ("learned_alt32_adamw_freeze80", "learned_alt8_adamw_freeze80"),
    ("learned_joint_adamw_cosine_geometry_decay", "learned_joint_adamw_no_freeze"),
)
ARM_ORDER = (
    "fixed_keys",
    "learned_joint_adamw_freeze80",
    "learned_joint_adamw_no_freeze",
    "learned_alt8_adamw_freeze80",
    "learned_joint_adamw_freeze25",
    "learned_joint_adamw_freeze50",
    "learned_alt32_adamw_freeze80",
    "learned_joint_adamw_cosine_geometry_decay",
)
COLORS = dict(zip(ARM_ORDER, ("#64748b", "#0f766e", "#a21caf", "#1d4ed8", "#0f9f8f", "#14b8a6", "#c2410c", "#7c3aed")))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _final_behavior(row: dict[str, Any]) -> dict[str, Any]:
    for point in reversed(row.get("traces", [])):
        if isinstance(point.get("behavior"), dict):
            return point["behavior"]
    return {}


def seed_grain_row(row: dict[str, Any]) -> dict[str, Any]:
    traces = [point for point in row.get("traces", []) if point.get("phase") != "freeze_event"]
    final = traces[-1] if traces else {}
    behavior = _final_behavior(row)
    routing = behavior.get("routing_decomposition", {}).get("states", {}).get("Qt_Kt", {})
    return {
        "row_id": row.get("row_id"),
        "arm": row.get("arm"),
        "seed": row.get("seed"),
        "test_loss": row.get("test_loss"),
        "validation_loss": row.get("validation_loss"),
        "test_perplexity": row.get("test_perplexity"),
        "relative_key_drift": final.get("groups", {}).get("memory_keys", {}).get("cumulative_relative_l2_delta_from_initial"),
        "postfreeze_relative_key_drift": row.get("postfreeze_relative_l2_drift"),
        "memory_contribution_ratio": behavior.get("memory_contribution_ratio_mean"),
        "memory_output_stable_rank": behavior.get("memory_output_stable_rank_mean"),
        "support_entropy": routing.get("normalized_support_entropy"),
        "dead_support_fraction": routing.get("dead_support_fraction"),
        "effective_support_count": routing.get("global_effective_support_count"),
        "tokens_per_second": row.get("tokens_per_second"),
        "peak_vram_mib": float(row.get("peak_vram_bytes", 0)) / 1048576,
        "freeze_fraction_observed": row.get("freeze_fraction_observed"),
        "final_geometry_learning_rate": row.get("final_geometry_learning_rate"),
    }


def _bootstrap_mean_ci(values: list[float], *, seed: int, draws: int = 20_000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = generator.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _paired_randomization_p(values: list[float], *, seed: int, monte_carlo_draws: int = 200_000) -> tuple[float, str, int]:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    n = len(array)
    if n <= 20:
        totals = 0
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float((array * np.asarray(signs)).mean()))
            totals += 1
            extreme += int(statistic >= observed - 1e-15)
        return extreme / totals, "exact_paired_sign_flip", totals
    generator = np.random.default_rng(seed)
    extreme = 0
    remaining = monte_carlo_draws
    while remaining:
        size = min(10_000, remaining)
        signs = generator.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        statistics = np.abs((signs * array).mean(axis=1))
        extreme += int(np.count_nonzero(statistics >= observed - 1e-15))
        remaining -= size
    return (extreme + 1) / (monte_carlo_draws + 1), "monte_carlo_paired_sign_flip", monte_carlo_draws


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["p_value_two_sided"]))
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * float(row["p_value_two_sided"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running
        rows[index]["reject_holm_0_05"] = running <= 0.05


def paired_comparisons(seed_rows: list[dict[str, Any]], comparisons: tuple[tuple[str, str], ...], family: str) -> list[dict[str, Any]]:
    lookup = {(str(row["arm"]), int(row["seed"])): row for row in seed_rows}
    output = []
    for index, (first, second) in enumerate(comparisons):
        shared = sorted({seed for arm, seed in lookup if arm == first} & {seed for arm, seed in lookup if arm == second})
        first_losses = [float(lookup[(first, seed)]["test_loss"]) for seed in shared]
        second_losses = [float(lookup[(second, seed)]["test_loss"]) for seed in shared]
        differences = [math.log(first_loss / second_loss) for first_loss, second_loss in zip(first_losses, second_losses)]
        mean = statistics.mean(differences)
        standard_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
        low, high = _bootstrap_mean_ci(differences, seed=61_000 + index + (0 if family == "primary" else 100))
        p_value, method, draws = _paired_randomization_p(differences, seed=62_000 + index + (0 if family == "primary" else 100))
        output.append({
            "family": family,
            "first_arm": first,
            "second_arm": second,
            "estimand": "paired log held-out-loss ratio log(first / second)",
            "direction": "negative and geometric_relative_change below zero favor first arm",
            "paired_seeds": shared,
            "n": len(shared),
            "mean_log_loss_ratio": mean,
            "geometric_relative_change": math.exp(mean) - 1.0,
            "median_log_loss_ratio": statistics.median(differences),
            "median_paired_relative_change": math.exp(statistics.median(differences)) - 1.0,
            "win_rate_first_lower_loss": sum(first_loss < second_loss for first_loss, second_loss in zip(first_losses, second_losses)) / len(shared),
            "bootstrap_95_ci_log_ratio": [low, high],
            "bootstrap_95_ci_relative_change": [math.exp(low) - 1.0, math.exp(high) - 1.0],
            "equivalence_margin_relative": 0.01,
            "equivalence_supported_by_95_ci": low > math.log(0.99) and high < math.log(1.01),
            "standardized_effect_dz": mean / standard_deviation if standard_deviation > 0 else None,
            "p_value_two_sided": p_value,
            "test": method,
            "randomization_draws": draws,
            "differences": differences,
        })
    _holm(output)
    return output


def _save(figure, root: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    return paths


def build_stage1_figures(results: list[dict[str, Any]], comparisons: list[dict[str, Any]], report_root: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = report_root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_arm[str(row["arm"])].append(row)

    figure, axis = plt.subplots(figsize=(11, 6))
    for arm in ARM_ORDER:
        token_values: dict[int, list[float]] = defaultdict(list)
        for row in by_arm.get(arm, []):
            for point in row.get("traces", []):
                if point.get("phase") != "freeze_event":
                    token_values[int(point["tokens"])].append(float(point["validation_loss"]))
        if not token_values:
            continue
        tokens = sorted(token_values)
        median = [float(np.median(token_values[token])) for token in tokens]
        low = [float(np.quantile(token_values[token], 0.25)) for token in tokens]
        high = [float(np.quantile(token_values[token], 0.75)) for token in tokens]
        x = [token / 1e6 for token in tokens]
        axis.fill_between(x, low, high, color=COLORS[arm], alpha=.10)
        axis.plot(x, median, marker="o", color=COLORS[arm], label=f"{arm} (n={len(by_arm[arm])})")
    axis.set(xlabel="Training tokens (millions)", ylabel="Validation cross-entropy", title="Stage 1 learning curves: median and interquartile range")
    axis.grid(alpha=.2); axis.legend(fontsize=7, ncol=2)
    paths += _save(figure, root, "stage1_learning_curves")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    for arm in ARM_ORDER:
        token_values: dict[int, list[float]] = defaultdict(list)
        for row in by_arm.get(arm, []):
            for point in row.get("traces", []):
                if point.get("phase") == "freeze_event":
                    continue
                value = point.get("groups", {}).get("memory_keys", {}).get("cumulative_relative_l2_delta_from_initial")
                if value is not None:
                    token_values[int(point["tokens"])].append(float(value))
        if token_values:
            tokens = sorted(token_values)
            axis.plot([token / 1e6 for token in tokens], [float(np.median(token_values[token])) for token in tokens], marker="o", color=COLORS[arm], label=arm)
    axis.set(xlabel="Training tokens (millions)", ylabel="Median relative key drift", title="Memory-geometry displacement through training")
    axis.set_yscale("symlog", linthresh=1e-8); axis.grid(alpha=.2); axis.legend(fontsize=7, ncol=2)
    paths += _save(figure, root, "stage1_key_drift")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    arm = "learned_joint_adamw_cosine_geometry_decay"
    token_values: dict[int, list[float]] = defaultdict(list)
    for row in by_arm.get(arm, []):
        for point in row.get("traces", []):
            value = point.get("learning_rates", {}).get("geometry")
            if value is not None:
                token_values[int(point["tokens"])].append(float(value))
    if token_values:
        tokens = sorted(token_values)
        axis.plot([token / 1e6 for token in tokens], [float(np.median(token_values[token])) for token in tokens], marker="o", color=COLORS[arm])
    axis.set(xlabel="Training tokens (millions)", ylabel="Geometry learning rate", title="Registered smooth geometry stabilization schedule")
    axis.grid(alpha=.2)
    paths += _save(figure, root, "stage1_cosine_geometry_lr")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    labels = [f"{row['first_arm']} −\n{row['second_arm']}" for row in comparisons]
    means = [100 * float(row["geometric_relative_change"]) for row in comparisons]
    lows = [100 * float(row["bootstrap_95_ci_relative_change"][0]) for row in comparisons]
    highs = [100 * float(row["bootstrap_95_ci_relative_change"][1]) for row in comparisons]
    positions = np.arange(len(comparisons))
    axis.errorbar(means, positions, xerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="o", color="#0f172a", capsize=4)
    axis.axvline(0, color="#94a3b8", linewidth=1)
    axis.set_yticks(positions, labels, fontsize=7)
    axis.set(xlabel="Geometric relative test-loss change, % (negative favors first arm)", title="Registered Stage 1 paired log-loss-ratio effects with bootstrap 95% intervals")
    axis.grid(axis="x", alpha=.2)
    paths += _save(figure, root, "stage1_paired_effects")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    values = [[float(row["test_loss"]) for row in by_arm.get(arm, [])] for arm in ARM_ORDER]
    axis.boxplot(values, tick_labels=ARM_ORDER, showmeans=True)
    axis.tick_params(axis="x", rotation=30, labelsize=7)
    axis.set(ylabel="Held-out test cross-entropy", title="Stage 1 held-out performance by registered arm")
    axis.grid(axis="y", alpha=.2)
    paths += _save(figure, root, "stage1_test_loss_distributions")
    plt.close(figure)
    return paths


def analyze_stage1(run_root: str | Path, report_root: str | Path, manifest: str | Path) -> dict[str, Any]:
    run_root = Path(run_root); report_root = Path(report_root); report_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = load_results(run_root)
    execution_audit = audit_results(results, manifest_rows)
    expected = {str(row["row_id"]) for row in manifest_rows}
    observed = {str(row.get("row_id")) for row in results}
    seed_rows = [seed_grain_row(row) for row in results if row.get("status") == "pass"]
    checks = {
        "all_168_rows_present": len(results) == 168 and observed == expected,
        "all_rows_passed": len(results) == 168 and all(row.get("status") == "pass" for row in results),
        "initial_state_identity": bool(execution_audit["checks"].get("initial_states_identical_within_seed_and_identity_group")),
        "anchor_identity_within_seed_and_size": bool(execution_audit["checks"].get("anchors_identical_within_seed_and_size")),
        "sample_order_identity": bool(execution_audit["checks"].get("sample_order_identical_within_seed_and_budget")),
        "finite_complete_metrics": bool(execution_audit["checks"].get("finite_metrics")),
        "optimizer_provenance": bool(execution_audit["checks"].get("executable_optimizer_provenance")),
        "trace_completeness": bool(execution_audit["checks"].get("standard_trace_complete")),
        "restart_identity": bool(execution_audit["checks"].get("restart_identity_when_registered")),
        "registered_anchor_size": bool(results) and all(int(row.get("anchor_token_states_resolved", 0)) == 16_384 for row in results),
        "finite_primary_outcome": len(seed_rows) == len(results) and all(math.isfinite(float(row["test_loss"])) for row in seed_rows),
        "primary_pairing_30_seeds": all(len({row["seed"] for row in seed_rows if row["arm"] == arm}) == 30 for arm in ARM_ORDER[:4]),
        "secondary_pairing_12_seeds": all(len({row["seed"] for row in seed_rows if row["arm"] == arm}) == 12 for arm in ARM_ORDER[4:]),
        "freeze_integrity": bool(execution_audit["checks"].get("freeze_integrity")),
        "cosine_schedule_recorded": all(row.get("geometry_lr_schedule") == "cosine" and float(row.get("final_geometry_learning_rate", math.inf)) < 1e-8 for row in results if row.get("arm") == "learned_joint_adamw_cosine_geometry_decay"),
        "semantic_permutation_identity": all(row.get("matched_key_expert_permutation", {}).get("passed", True) for row in results),
        "bf16_operational_gate": all(row.get("matched_key_expert_permutation", {}).get("operational_within_expected_precision_tolerance", True) for row in results),
    }
    primary = paired_comparisons(seed_rows, PRIMARY_COMPARISONS, "primary") if checks["primary_pairing_30_seeds"] else []
    secondary = paired_comparisons(seed_rows, SECONDARY_COMPARISONS, "secondary") if checks["secondary_pairing_12_seeds"] and checks["primary_pairing_30_seeds"] else []
    comparison_rows = primary + secondary
    figures = build_stage1_figures(results, comparison_rows, report_root) if results and comparison_rows else []
    decision = "STAGE1_COMPLETE" if all(checks.values()) else "STAGE1_INCOMPLETE"
    summary = {
        "campaign": "phase6_behavioral_atlas_v2",
        "stage": "stage1_core_lifecycle",
        "decision": decision,
        "checks": checks,
        "expected_rows": len(manifest_rows),
        "observed_rows": len(results),
        "missing_row_ids": sorted(expected - observed),
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "figures": figures,
        "multiplicity": "Holm within primary and secondary families",
        "primary_transform": "paired_log_loss_ratio",
        "uncertainty": "20,000-draw paired bootstrap 95% CI; exact or 200,000-draw paired sign-flip randomization test",
    }
    _atomic_json(run_root / "behavioral_atlas_stage1_summary.json", summary)
    grain_path = report_root / "stage1_seed_grain_metrics.jsonl"
    grain_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in seed_rows), encoding="utf-8")
    comparison_path = report_root / "stage1_paired_comparisons.jsonl"
    comparison_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in comparison_rows), encoding="utf-8")
    lines = [
        "# Phase 6.2 Stage 1 Core Lifecycle Report", "", f"**Decision:** `{decision}`", "",
        f"Completed rows: {len(results)}/{len(manifest_rows)}. Primary transform is the paired log held-out-loss ratio; negative geometric relative changes favor the first named arm.", "",
        "## Validity checks", "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    lines.extend(["", "## Registered paired comparisons", "", "| Family | First ÷ second | n | Geometric relative Δ | Bootstrap 95% CI | Win rate | dz | raw p | Holm p | Reject |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in comparison_rows:
        ci = row["bootstrap_95_ci_relative_change"]
        dz = "—" if row["standardized_effect_dz"] is None else f"{row['standardized_effect_dz']:.3g}"
        lines.append(f"| {row['family']} | {row['first_arm']} ÷ {row['second_arm']} | {row['n']} | {100 * row['geometric_relative_change']:.3g}% | [{100 * ci[0]:.3g}%, {100 * ci[1]:.3g}%] | {row['win_rate_first_lower_loss']:.3g} | {dz} | {row['p_value_two_sided']:.4g} | {row['holm_adjusted_p']:.4g} | {row['reject_holm_0_05']} |")
    lines.extend([
        "", "## What the figures show", "",
        "Learning curves show when arms separate; key-drift curves show geometry motion and freezing; the cosine-LR panel verifies smooth stabilization; paired-effect intervals show uncertainty; held-out distributions show seed variability.", "",
        "## Interpretation boundary", "",
        "Only the registered paired comparisons support confirmatory claims. Secondary comparisons are Holm-corrected as a separate family. Stage 1 does not select the broad Stage 2 solution space without a documented gate review.",
    ])
    (report_root / "BEHAVIORAL_ATLAS_STAGE1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    significant = [row for row in comparison_rows if row.get("reject_holm_0_05")]
    handoff = [
        "# LLM handoff: Phase 6.2 Stage 1", "", "## Ask", "",
        "Review the paired lifecycle evidence and advise whether to proceed to the registered constrained solution-space screen, modify a mechanism, or stop an arm.", "",
        "## Status", "", f"- Decision: `{decision}`", f"- Rows: {len(results)}/{len(manifest_rows)}", f"- Holm-significant registered comparisons: {len(significant)}/{len(comparison_rows)}", "",
        "## Evidence files", "", "- `BEHAVIORAL_ATLAS_STAGE1_REPORT.md`", "- `stage1_seed_grain_metrics.jsonl`", "- `stage1_paired_comparisons.jsonl`", "- `behavioral_atlas_stage1_summary.json` in the run root", "",
        "## Constraints", "", "Treat Stage 0 as measurement-only. Respect paired seeds, separate primary/secondary Holm families, effect intervals, parameter-drift diagnostics, and the fixed 50M-token budget.",
    ]
    (report_root / "BEHAVIORAL_ATLAS_STAGE1_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    summary = analyze_stage1(args.run_root, args.report_root, args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["decision"] != "STAGE1_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
