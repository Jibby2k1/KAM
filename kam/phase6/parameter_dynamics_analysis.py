"""Audit, statistics, reports, and figures for Phase 6.1 parameter dynamics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kam.phase6.parameter_dynamics_manifest import ARMS
from kam.phase6.parameter_dynamics_statistics import bootstrap_mean, holm_adjust, paired_log_comparison

COLORS = {"fixed_keys": "#475569", "learned_joint_freeze50": "#c2410c", "learned_joint_freeze80": "#0f766e",
          "learned_alt8_freeze80": "#1d4ed8", "learned_joint_no_freeze": "#a21caf"}
LINESTYLES = {"fixed_keys": "--", "learned_joint_freeze50": ":", "learned_joint_freeze80": "-",
              "learned_alt8_freeze80": "-.", "learned_joint_no_freeze": (0, (5, 2))}


def load_results(run_root: str | Path) -> list[dict[str, Any]]:
    row_root = Path(run_root) / "rows" / "parameter_dynamics_v1"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(row_root.glob("*.json"))]


def _event_audit(row: dict[str, Any]) -> dict[str, bool]:
    traces = row.get("traces", []); fixed = row.get("arm") == "fixed_keys"; freeze_fraction = float(row.get("freeze_fraction", 1.0))
    event_indices = [index for index, point in enumerate(traces) if point.get("phase") == "freeze_event"]
    event_order = fixed or freeze_fraction >= 1.0 or (len(event_indices) == 1 and all(point.get("phase") != "post_freeze" for point in traces[:event_indices[0]])
                                                       and all(point.get("phase") == "post_freeze" for point in traces[event_indices[0] + 1:]))
    return {"event_order_valid": event_order,
            "postfreeze_hash_unchanged": fixed or freeze_fraction >= 1.0 or bool(row.get("postfreeze_key_hash_unchanged")),
            "postfreeze_drift_valid": fixed or freeze_fraction >= 1.0 or float(row.get("postfreeze_relative_l2_drift", math.inf)) <= 1e-12,
            "no_postfreeze_key_gradient": not bool(row.get("postfreeze_key_grad_observed"))}


def audit_results(results: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {row["row_id"] for row in manifest_rows}; observed = [row.get("row_id") for row in results]
    initial_hashes: dict[int, set[str]] = defaultdict(set)
    for row in results: initial_hashes[int(row["seed"])].add(str(row.get("initial_state_hash")))
    event_rows = [{"row_id": row.get("row_id"), **_event_audit(row)} for row in results]
    checks = {"all_rows_present": set(observed) == expected and len(observed) == len(expected),
              "all_rows_passed": len(results) == len(manifest_rows) and all(row.get("status") == "pass" for row in results),
              "initial_states_identical_within_seed": bool(initial_hashes) and all(len(values) == 1 for values in initial_hashes.values()),
              "finite_metrics": bool(results) and all(np.isfinite(float(row.get("test_loss", math.nan))) for row in results),
              "all_parameter_groups_present": bool(results) and all(set(point.get("groups", {})) == {"memory_keys", "memory_experts", "memory_gates", "attention", "feedforward", "embeddings", "output_head"}
                                                                    for row in results for point in row.get("traces", [])),
              "fixed_keys_unchanged": all(row.get("initial_key_hash") == row.get("final_key_hash") for row in results if row.get("arm") == "fixed_keys"),
              "freeze_event_order_valid": all(row["event_order_valid"] for row in event_rows),
              "freeze_integrity": all(row["postfreeze_hash_unchanged"] and row["postfreeze_drift_valid"] and row["no_postfreeze_key_gradient"] for row in event_rows)}
    return {"passed": all(checks.values()), "checks": checks, "expected_rows": len(manifest_rows), "observed_rows": len(results),
            "missing_row_ids": sorted(expected - set(observed)), "event_rows": event_rows}


def _bootstrap(values: list[float], seed: int = 6610) -> tuple[float, float]:
    if not values: return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    if len(array) == 1: return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed); draws = rng.choice(array, size=(20_000, len(array)), replace=True).mean(1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _rates(row: dict[str, Any]) -> list[tuple[int, float, str]]:
    return [(int(point["tokens"]), float(point["groups"]["memory_keys"]["incremental_relative_delta_per_million_tokens"]), str(point["phase"]))
            for point in row.get("traces", []) if int(point.get("tokens", 0)) > 0 and point.get("phase") != "freeze_event"]


def stabilization_ratio(row: dict[str, Any]) -> float | None:
    rates = _rates(row); pre = [(tokens, value) for tokens, value, phase in rates if phase == "pre_freeze" and value > 0]
    if len(pre) < 2: return None
    early = [value for tokens, value in pre if 1_000_000 <= tokens <= 10_000_000] or [pre[0][1]]
    late = [value for tokens, value in pre if 30_000_000 <= tokens <= 40_000_000] or [pre[-1][1]]
    return float(np.mean(late) / max(float(np.mean(early)), 1e-30))


def _series(results, arm: str, value: Callable[[dict[str, Any]], float]) -> tuple[list[int], list[float]]:
    by_tokens: dict[int, list[float]] = defaultdict(list)
    for row in results:
        if row.get("arm") != arm: continue
        for point in row.get("traces", []):
            if point.get("phase") == "freeze_event": continue
            by_tokens[int(point["tokens"])].append(value(point))
    tokens = sorted(by_tokens); return tokens, [float(np.median(by_tokens[token])) for token in tokens]


def _save(figure, root: Path, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = root / f"{name}.{suffix}"; figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight"); paths.append(str(path))
    return paths


def build_figures(results: list[dict[str, Any]], report_root: str | Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(report_root) / "figures"; root.mkdir(parents=True, exist_ok=True); paths: list[str] = []
    groups = ["memory_keys", "memory_experts", "memory_gates", "attention", "feedforward", "embeddings", "output_head"]
    figure, axes = plt.subplots(4, 2, figsize=(13, 14)); axes = axes.flatten()
    for axis, group in zip(axes, groups):
        for arm in ARMS:
            x, y = _series(results, arm, lambda p, g=group: float(p["groups"][g]["cumulative_relative_l2_delta_from_initial"]))
            if x: axis.plot(np.asarray(x) / 1e6, y, label=arm, color=COLORS[arm], linestyle=LINESTYLES[arm])
        axis.set_title(group.replace("_", " ").title()); axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel("Relative L2 drift"); axis.grid(alpha=.2)
    axes[-1].axis("off"); axes[0].legend(fontsize=7); figure.suptitle("Parameter-group displacement from initialization"); paths += _save(figure, root, "parameter_group_relative_drift"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 6))
    for arm in ARMS:
        x, y = _series(results, arm, lambda p: max(float(p["groups"]["memory_keys"]["incremental_relative_delta_per_million_tokens"]), 1e-12))
        if x: axis.plot(np.asarray(x) / 1e6, y, label=arm, color=COLORS[arm], linestyle=LINESTYLES[arm], marker="o", ms=3)
    axis.set_yscale("log"); axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel("Relative key update per 1M tokens (log)"); axis.set_title("Memory-key update rate and freeze schedule"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    paths += _save(figure, root, "key_update_rate_and_freeze"); plt.close(figure)
    selected = [row for row in results if row.get("arm") == "learned_joint_freeze80"] or results[:1]
    token_values = sorted({int(point["tokens"]) for row in selected for point in row.get("traces", []) if point.get("phase") != "freeze_event"})
    layer_count = max((len(point["memory"].get("layer_relative_key_drift", [])) for row in selected for point in row.get("traces", [])), default=1)
    matrix = np.full((layer_count, len(token_values)), np.nan)
    for col, token in enumerate(token_values):
        values = [point["memory"]["layer_relative_key_drift"] for row in selected for point in row.get("traces", []) if int(point["tokens"]) == token and point.get("phase") != "freeze_event"]
        if values: matrix[:, col] = np.mean(np.asarray(values), axis=0)
    figure, axis = plt.subplots(figsize=(11, 5)); image = axis.imshow(np.log10(np.maximum(matrix, 1e-12)), aspect="auto", cmap="Blues")
    axis.set_title("Layerwise memory-key drift — joint freeze-80"); axis.set_xlabel("Checkpoint tokens (millions)"); axis.set_ylabel("Memory layer"); axis.set_xticks(range(len(token_values)), [f"{v/1e6:g}" for v in token_values]); figure.colorbar(image, ax=axis, label="log10 relative drift")
    paths += _save(figure, root, "layer_checkpoint_drift_heatmap"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 6))
    for arm in ARMS:
        x, y = _series(results, arm, lambda p: float(p["memory"]["key_angular_displacement_median"]))
        if x: axis.plot(np.asarray(x)/1e6, y, label=arm, color=COLORS[arm], linestyle=LINESTYLES[arm], marker="o", ms=3)
    axis.set_title("Median memory-key angular displacement"); axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel("Radians"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    paths += _save(figure, root, "key_angular_displacement"); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm in ARMS:
        for axis, field, label in ((axes[0], "routing_topk_jaccard_to_initial", "Top-k Jaccard to initialization"), (axes[1], "support_usage_entropy", "Normalized support-use entropy")):
            x, y = _series(results, arm, lambda p, f=field: float(p["memory"][f]));
            if x: axis.plot(np.asarray(x)/1e6, y, label=arm, color=COLORS[arm], linestyle=LINESTYLES[arm])
            axis.set_xlabel("Training tokens (millions)"); axis.set_ylabel(label); axis.grid(alpha=.2)
    axes[0].set_title("Routing identity retention"); axes[1].set_title("Support usage"); axes[0].legend(fontsize=7)
    paths += _save(figure, root, "routing_stability_and_usage"); plt.close(figure)
    ratio_by_arm = {arm: [ratio for row in results if row.get("arm") == arm and (ratio := stabilization_ratio(row)) is not None] for arm in ARMS if arm != "fixed_keys"}
    figure, axis = plt.subplots(figsize=(10, 5)); labels = list(ratio_by_arm)
    for index, arm in enumerate(labels):
        values = ratio_by_arm[arm]; center = float(np.mean(values)) if values else math.nan; low, high = _bootstrap(values)
        if np.isfinite(center): axis.errorbar(index, center, yerr=[[center-low], [high-center]], fmt="o", color=COLORS[arm], capsize=4)
    axis.axhline(.25, color="#334155", linestyle="--", label="Practical threshold 0.25"); axis.set_xticks(range(len(labels)), [label.replace("learned_", "") for label in labels], rotation=15); axis.set_ylabel("Late / early key update rate"); axis.set_title("Natural key-stabilization ratio"); axis.grid(axis="y", alpha=.2); axis.legend()
    paths += _save(figure, root, "stabilization_ratio_intervals"); plt.close(figure)
    fixed = {(int(row["seed"]), int(point["tokens"])): float(point["validation_loss"]) for row in results if row.get("arm") == "fixed_keys" for point in row.get("traces", [])}
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in results:
        if row.get("arm") == "fixed_keys": continue
        candidates = [point for point in row.get("traces", []) if point.get("phase") == "pre_freeze"] or row.get("traces", [])[-1:]
        if not candidates: continue
        point = candidates[-1]; baseline = fixed.get((int(row["seed"]), int(point["tokens"])))
        if baseline is None: continue
        x = float(point["groups"]["memory_keys"]["cumulative_relative_l2_delta_from_initial"]); y = baseline - float(point["validation_loss"])
        axis.scatter(x, y, color=COLORS[row["arm"]], alpha=.75, label=row["arm"])
    handles, labels = axis.get_legend_handles_labels(); unique = dict(zip(labels, handles)); axis.legend(unique.values(), unique.keys(), fontsize=8)
    axis.axhline(0, color="#334155", linewidth=1); axis.set_xlabel("Relative key drift from initialization"); axis.set_ylabel("Validation improvement vs paired fixed keys"); axis.set_title("Parameter change and validation change"); axis.grid(alpha=.2)
    paths += _save(figure, root, "parameter_change_vs_validation_change"); plt.close(figure)
    freeze = {int(row["seed"]): float(row["test_loss"]) for row in results if row.get("arm") == "learned_joint_freeze80"}; nofreeze = {int(row["seed"]): float(row["test_loss"]) for row in results if row.get("arm") == "learned_joint_no_freeze"}
    figure, axis = plt.subplots(figsize=(8, 6))
    for seed in sorted(freeze.keys() & nofreeze.keys()): axis.plot([0, 1], [freeze[seed], nofreeze[seed]], color="#94a3b8", alpha=.6); axis.scatter([0,1], [freeze[seed], nofreeze[seed]], color=[COLORS["learned_joint_freeze80"], COLORS["learned_joint_no_freeze"]])
    axis.set_xticks([0,1], ["Freeze at 80%", "No freeze"]); axis.set_ylabel("Held-out test cross-entropy"); axis.set_title("Final-tuning freeze effect by paired seed"); axis.grid(axis="y", alpha=.2)
    paths += _save(figure, root, "final_tuning_freeze_effect"); plt.close(figure)
    return paths

def analyze_parameter_dynamics(run_root: str | Path, report_root: str | Path, manifest: str | Path) -> dict[str, Any]:
    run_root, report_root = Path(run_root), Path(report_root); report_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = load_results(run_root); audit = audit_results(results, manifest_rows); figures = build_figures(results, report_root)
    ratios = {arm: [value for row in results if row.get("arm") == arm and (value := stabilization_ratio(row)) is not None] for arm in ARMS}
    stabilization = {}
    for arm, values in ratios.items():
        low, high = bootstrap_mean(values)
        stabilization[arm] = {"seeds": len(values), "mean_ratio": float(np.mean(values)) if values else None,
                              "ci_low": low if values else None, "ci_high": high if values else None,
                              "nearly_frozen_pass": bool(values and high < .25)}
    learned_geometry = [paired_log_comparison(results, arm, "fixed_keys", metric="validation_loss", checkpoint=40_000_000)
                        for arm in ("learned_joint_freeze80", "learned_alt8_freeze80")]
    complete_geometry = [comparison for comparison in learned_geometry if comparison.get("paired_seeds")]
    adjusted = holm_adjust({comparison["candidate"]: float(comparison["paired_sign_flip_p"]) for comparison in complete_geometry})
    for comparison in learned_geometry: comparison["holm_adjusted_p"] = adjusted.get(comparison["candidate"])
    freeze_effect = paired_log_comparison(results, "learned_joint_freeze80", "learned_joint_no_freeze", metric="test_loss")
    stage = str(manifest_rows[0].get("stage")) if manifest_rows else "unknown"
    decision = "PILOT_PASS" if stage == "pilot" and audit["passed"] else "PILOT_BLOCKED" if stage == "pilot" else "MAIN_COMPLETE" if audit["passed"] else "MAIN_BLOCKED"
    summary = {"campaign": "phase6_parameter_dynamics_v1", "stage": stage, "decision": decision, "audit": audit,
               "stabilization_ratios": ratios, "stabilization": stabilization,
               "learned_geometry_comparisons": learned_geometry, "final_tuning_freeze_effect": freeze_effect,
               "figures": figures, "rows": len(results)}
    (run_root / "parameter_dynamics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Phase 6.1 Parameter-Dynamics Report", "", f"**Stage:** `{stage}`", f"**Decision:** `{decision}`", "",
             f"- Complete rows: {len(results)}/{len(manifest_rows)}", f"- Audit passed: `{audit['passed']}`",
             f"- Required figures: {len(figures)//2}/8 (PNG and SVG)", "", "## Audit", ""]
    lines += [f"- {key}: `{value}`" for key, value in audit["checks"].items()]
    lines += ["", "## Locked estimands", ""]
    for arm, result in stabilization.items():
        lines.append(f"- {arm}: stabilization ratio `{result['mean_ratio']}`; 95% CI `[{result['ci_low']}, {result['ci_high']}]`; nearly-frozen pass `{result['nearly_frozen_pass']}`.")
    for comparison in learned_geometry:
        lines.append(f"- {comparison['candidate']} vs fixed keys at 40M: relative validation change `{comparison.get('geometric_relative_change')}`; Holm p `{comparison.get('holm_adjusted_p')}`.")
    lines.append(f"- Freeze-80 vs no-freeze at final test: relative change `{freeze_effect.get('geometric_relative_change')}`; paired p `{freeze_effect.get('paired_sign_flip_p')}`.")
    lines += ["", "## Interpretation boundary", "", "Pilot rows validate instrumentation only. Main rows test parameter stabilization and freeze effects; neither stage can overturn confirmation v2."]
    (report_root / "PARAMETER_DYNAMICS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


__all__ = ["analyze_parameter_dynamics", "audit_results", "build_figures", "load_results", "stabilization_ratio"]
