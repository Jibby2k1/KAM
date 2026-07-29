"""Locked statistics, guardrails, figures, and decision for confirmation v2."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .confirmation_manifest import EXPECTED_ROWS, MECHANISM_SEEDS, PRIMARY_SEEDS, REPLICATION_SEEDS, TARGET_TOKENS
from .stats import holm_adjust


PRIMARY_MARGIN = math.log(0.98)
ALPHA = 0.05
BOOTSTRAP_REPLICATES = 20_000
MONTE_CARLO_PERMUTATIONS = 100_000


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _load_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "rows" / "confirmation_v2").glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _observations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        subruns = metrics.get("subruns", []) if isinstance(metrics, dict) else []
        if row.get("status") != "pass" or len(subruns) != 1:
            continue
        run = subruns[0]
        output.append(
            {
                "row_id": row.get("row_id"),
                "cohort": row.get("cohort"),
                "corpus_id": row.get("corpus_id"),
                "architecture": row.get("architecture"),
                "training_seed": int(run.get("training_seed", row.get("seed"))),
                "data_seed": int(row.get("data_seed")),
                "test_loss": run.get("test_loss"),
                "validation_loss": run.get("validation_loss"),
                "best_validation_loss": run.get("best_validation_loss"),
                "generalization_gap": run.get("generalization_gap"),
                "tokens": run.get("tokens"),
                "target_tokens": run.get("target_tokens_resolved"),
                "wall_seconds": run.get("wall_seconds"),
                "tokens_per_second": run.get("tokens_per_second"),
                "estimated_training_flops": run.get("estimated_training_flops"),
                "active_parameters_per_token": run.get("active_parameters_per_token"),
                "total_parameters": run.get("total_parameters"),
                "peak_vram_bytes": run.get("peak_vram_bytes"),
                "dataset_sha256": run.get("dataset_sha256"),
                "train_sha256": run.get("train_sha256"),
                "validation_sha256": run.get("validation_sha256"),
                "test_sha256": run.get("test_sha256"),
                "geometry_steps": run.get("geometry_steps"),
                "geometry_freeze_tokens": run.get("geometry_freeze_tokens"),
                "geometry_frozen_for_final_tuning": run.get("geometry_frozen_for_final_tuning"),
                "post_freeze_geometry_drift": run.get("post_freeze_geometry_drift"),
                "loss_history": run.get("loss_history", []),
                "deletion_metrics": run.get("deletion_metrics", []),
            }
        )
    return output


def _bootstrap_mean(values: np.ndarray, *, seed: int = 6606) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, values.size), replace=True)
    estimates = samples.mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _sign_flip_p(values: np.ndarray, *, seed: int = 6606) -> tuple[float, int, str]:
    observed = abs(float(values.mean()))
    if values.size <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=values.size)), dtype=float)
        null = np.mean(signs * values, axis=1)
        return float(np.mean(np.abs(null) >= observed)), int(signs.shape[0]), "exact"
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(MONTE_CARLO_PERMUTATIONS, values.size))
    null = np.mean(signs * values, axis=1)
    exceedances = int(np.sum(np.abs(null) >= observed))
    return (exceedances + 1) / (MONTE_CARLO_PERMUTATIONS + 1), MONTE_CARLO_PERMUTATIONS, "monte_carlo"


def paired_log_ratio(
    observations: list[dict[str, Any]],
    *,
    corpus_id: str,
    candidate: str,
    comparator: str,
    seeds: Iterable[int],
    comparison: str,
) -> dict[str, Any]:
    seed_set = {int(seed) for seed in seeds}
    candidate_values = {
        int(row["training_seed"]): float(row["test_loss"])
        for row in observations
        if row.get("corpus_id") == corpus_id
        and row.get("architecture") == candidate
        and int(row["training_seed"]) in seed_set
        and _finite(row.get("test_loss"))
    }
    comparator_values = {
        int(row["training_seed"]): float(row["test_loss"])
        for row in observations
        if row.get("corpus_id") == corpus_id
        and row.get("architecture") == comparator
        and int(row["training_seed"]) in seed_set
        and _finite(row.get("test_loss"))
    }
    paired_seeds = sorted(candidate_values.keys() & comparator_values.keys())
    if not paired_seeds:
        return {
            "comparison": comparison,
            "candidate": candidate,
            "comparator": comparator,
            "corpus_id": corpus_id,
            "paired_seeds": 0,
            "seed_ids": [],
            "complete": False,
        }
    candidate_array = np.asarray([candidate_values[seed] for seed in paired_seeds], dtype=float)
    comparator_array = np.asarray([comparator_values[seed] for seed in paired_seeds], dtype=float)
    log_ratios = np.log(candidate_array / comparator_array)
    ci_low, ci_high = _bootstrap_mean(log_ratios)
    p_value, permutations, permutation_mode = _sign_flip_p(log_ratios)
    return {
        "comparison": comparison,
        "candidate": candidate,
        "comparator": comparator,
        "corpus_id": corpus_id,
        "paired_seeds": len(paired_seeds),
        "seed_ids": paired_seeds,
        "complete": paired_seeds == sorted(seed_set),
        "candidate_mean_test_loss": float(candidate_array.mean()),
        "comparator_mean_test_loss": float(comparator_array.mean()),
        "mean_log_ratio": float(log_ratios.mean()),
        "geometric_relative_change": float(math.exp(float(log_ratios.mean())) - 1.0),
        "ci_low_relative_change": float(math.exp(ci_low) - 1.0),
        "ci_high_relative_change": float(math.exp(ci_high) - 1.0),
        "bootstrap_ci_low_log_ratio": ci_low,
        "bootstrap_ci_high_log_ratio": ci_high,
        "paired_randomization_p": p_value,
        "permutations": permutations,
        "permutation_mode": permutation_mode,
        "median_relative_change": float(np.median(candidate_array / comparator_array - 1.0)),
        "candidate_win_rate": float(np.mean(candidate_array < comparator_array)),
        "standardized_effect_dz": float(log_ratios.mean() / log_ratios.std(ddof=1)) if log_ratios.size > 1 and log_ratios.std(ddof=1) > 0 else float("inf"),
        "candidate_values": candidate_array.tolist(),
        "comparator_values": comparator_array.tolist(),
    }


def _mechanism_audit(observations: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if observation.get("cohort") != "mechanism" or observation.get("architecture") not in {"T-KAM-L", "T-KAM-ALT"}:
            continue
        tokens = float(observation.get("tokens") or 0.0)
        freeze_tokens = float(observation.get("geometry_freeze_tokens") or 0.0)
        history = observation.get("loss_history", [])
        prefreeze_gradients = [
            float(point.get("memory_key_grad_norm", 0.0))
            for point in history
            if float(point.get("tokens", 0.0)) < freeze_tokens
        ]
        postfreeze_points = [point for point in history if float(point.get("tokens", 0.0)) >= freeze_tokens]
        checks = {
            "geometry_updated": float(observation.get("geometry_steps") or 0.0) > 0,
            "prefreeze_gradient_observed": max(prefreeze_gradients, default=0.0) > 0.0,
            "freeze_fraction_valid": tokens > 0 and 0.79 <= freeze_tokens / tokens <= 0.81,
            "frozen_final_tuning": bool(observation.get("geometry_frozen_for_final_tuning")),
            "zero_postfreeze_drift": abs(float(observation.get("post_freeze_geometry_drift") or 0.0)) <= 1e-10,
            "postfreeze_checkpoint_observed": bool(postfreeze_points) and all(float(point.get("geometry_frozen", 0.0)) == 1.0 for point in postfreeze_points),
        }
        rows.append(
            {
                "architecture": observation["architecture"],
                "training_seed": observation["training_seed"],
                "freeze_fraction": freeze_tokens / tokens if tokens else None,
                "max_prefreeze_key_gradient": max(prefreeze_gradients, default=0.0),
                "post_freeze_geometry_drift": observation.get("post_freeze_geometry_drift"),
                **checks,
                "passed": all(checks.values()),
            }
        )
    expected = 2 * len(MECHANISM_SEEDS)
    return {
        "expected_rows": expected,
        "observed_rows": len(rows),
        "passed_rows": sum(bool(row["passed"]) for row in rows),
        "passed": len(rows) == expected and all(bool(row["passed"]) for row in rows),
        "rows": rows,
    }


def _guardrails(rows: list[dict[str, Any]], observations: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {str(row["row_id"]) for row in manifest_rows}
    observed_ids = [str(row.get("row_id")) for row in rows]
    failures = [row for row in rows if row.get("status") != "pass"]
    corpus_hashes: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        if row.get("dataset_sha256"):
            corpus_hashes[str(row["corpus_id"])].add(str(row["dataset_sha256"]))
    pair_data_seeds: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in manifest_rows:
        pair_data_seeds[(str(row["corpus_id"]), int(row["seed"]))].add(int(row["data_seed"]))
    checks = {
        "manifest_row_count": len(manifest_rows) == EXPECTED_ROWS,
        "all_rows_present": set(observed_ids) == expected_ids,
        "no_duplicate_rows": len(observed_ids) == len(set(observed_ids)),
        "all_rows_passed": not failures and len(rows) == EXPECTED_ROWS,
        "one_dataset_hash_per_corpus": bool(corpus_hashes) and all(len(values) == 1 for values in corpus_hashes.values()),
        "paired_data_order": all(len(values) == 1 for values in pair_data_seeds.values()),
        "fixed_token_budget": len(observations) == EXPECTED_ROWS and all(
            _finite(row.get("tokens")) and TARGET_TOKENS <= float(row["tokens"]) < TARGET_TOKENS + 2048
            for row in observations
        ),
        "finite_test_metrics": len(observations) == EXPECTED_ROWS and all(_finite(row.get("test_loss")) and float(row["test_loss"]) > 0 for row in observations),
        "parameter_budget": len(observations) == EXPECTED_ROWS and all(
            _finite(row.get("total_parameters")) and abs(float(row["total_parameters"]) - 10_000_000) / 10_000_000 <= 0.05
            for row in observations
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_row_ids": sorted(expected_ids - set(observed_ids)),
        "unexpected_row_ids": sorted(set(observed_ids) - expected_ids),
        "failure_counts": dict(Counter(str(row.get("failure_category")) for row in failures)),
        "corpus_hash_counts": {key: len(values) for key, values in sorted(corpus_hashes.items())},
    }


def evaluate_confirmation(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = _observations(rows)
    primary = paired_log_ratio(
        observations,
        corpus_id="tinystories_v2_128mib",
        candidate="T-KAM-F",
        comparator="T-WIDE",
        seeds=PRIMARY_SEEDS,
        comparison="primary_tkamf_vs_twide",
    )
    replication = paired_log_ratio(
        observations,
        corpus_id="tinyshakespeare",
        candidate="T-KAM-F",
        comparator="T-WIDE",
        seeds=REPLICATION_SEEDS,
        comparison="replication_tkamf_vs_twide",
    )
    secondary = [
        paired_log_ratio(
            observations,
            corpus_id="tinystories_v2_128mib",
            candidate="T-KAM-F",
            comparator=comparator,
            seeds=PRIMARY_SEEDS[:12],
            comparison=f"secondary_tkamf_vs_{comparator.lower().replace('-', '')}",
        )
        for comparator in ("T0", "T-PKM")
    ]
    complete_secondary = [row for row in secondary if row.get("paired_seeds")]
    adjusted = holm_adjust({row["comparison"]: float(row["paired_randomization_p"]) for row in complete_secondary}) if complete_secondary else {}
    for row in secondary:
        row["holm_adjusted_p"] = adjusted.get(row["comparison"])
    mechanism = _mechanism_audit(observations)
    guardrails = _guardrails(rows, observations, manifest_rows)
    primary_pass = bool(
        primary.get("complete")
        and primary.get("paired_seeds") == len(PRIMARY_SEEDS)
        and float(primary.get("bootstrap_ci_high_log_ratio", math.inf)) <= PRIMARY_MARGIN
        and float(primary.get("paired_randomization_p", 1.0)) <= ALPHA
    )
    replication_pass = bool(
        replication.get("complete")
        and replication.get("paired_seeds") == len(REPLICATION_SEEDS)
        and float(replication.get("bootstrap_ci_high_log_ratio", math.inf)) < 0.0
        and float(replication.get("paired_randomization_p", 1.0)) <= ALPHA
    )
    if guardrails["passed"] and primary_pass and replication_pass:
        decision = "PROMOTE_FIXED_KEY_FAST_ALGEBRA"
        rationale = "T-KAM-F cleared the prespecified 2% TinyStories superiority margin and independently replicated on Tiny Shakespeare."
    elif not guardrails["passed"]:
        decision = "BLOCKED_INVALID_CONFIRMATION"
        rationale = "One or more preregistered completeness, comparability, or validity guardrails failed."
    else:
        decision = "RETAIN_AS_DIAGNOSTIC_ONLY"
        rationale = "The fixed-size primary and replication gates were not both satisfied; no post-hoc seed extension is permitted."
    return {
        "analysis_version": "phase6_confirmation_v2_locked",
        "decision": decision,
        "rationale": rationale,
        "primary_pass": primary_pass,
        "replication_pass": replication_pass,
        "learned_memory_lifecycle_pass": mechanism["passed"],
        "primary": primary,
        "replication": replication,
        "secondary": secondary,
        "mechanism": mechanism,
        "guardrails": guardrails,
        "observations": observations,
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{key: value for key, value in row.items() if key not in {"loss_history", "deletion_metrics", "candidate_values", "comparator_values"}} for row in rows]
    pq.write_table(pa.Table.from_pylist(serializable), path)
    return str(path)


def _build_figures(result: dict[str, Any], report_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    observations = result["observations"]
    paths: list[str] = []

    primary = result["primary"]
    figure, axis = plt.subplots(figsize=(9, 6))
    for index, (candidate, comparator) in enumerate(zip(primary.get("candidate_values", []), primary.get("comparator_values", []))):
        axis.plot([0, 1], [comparator, candidate], color="#94a3b8", alpha=0.45, linewidth=0.8)
        axis.scatter([0, 1], [comparator, candidate], color=["#475569", "#0f766e"], s=22)
    axis.set_xticks([0, 1], ["T-WIDE", "T-KAM-F"])
    axis.set_ylabel("Held-out test cross-entropy at 50M tokens")
    axis.set_title("Primary paired-seed confirmation — TinyStories")
    axis.grid(axis="y", alpha=0.25)
    path = figure_root / "primary_paired_test_loss.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    comparisons = [result["primary"], result["replication"], *result["secondary"]]
    complete = [row for row in comparisons if row.get("paired_seeds")]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(complete))
    centers = np.asarray([100 * float(row["geometric_relative_change"]) for row in complete])
    lows = np.asarray([100 * float(row["ci_low_relative_change"]) for row in complete])
    highs = np.asarray([100 * float(row["ci_high_relative_change"]) for row in complete])
    axis.errorbar(centers, y, xerr=[centers - lows, highs - centers], fmt="o", color="#0f766e", capsize=4)
    axis.axvline(0, color="#334155", linewidth=1)
    axis.axvline(-2, color="#be123c", linestyle="--", linewidth=1, label="Primary practical margin")
    axis.set_yticks(y, [str(row["comparison"]) for row in complete])
    axis.set_xlabel("T-KAM-F relative test-loss change (%)")
    axis.set_title("Preregistered paired effects with bootstrap 95% intervals")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    path = figure_root / "effect_intervals.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, cohort in zip(axes, ("primary", "replication")):
        grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in observations:
            if row.get("cohort") != cohort:
                continue
            for point in row.get("loss_history", []):
                if _finite(point.get("tokens")) and _finite(point.get("validation_loss")):
                    grouped[str(row["architecture"])][float(point["tokens"])].append(float(point["validation_loss"]))
        for architecture, points in sorted(grouped.items()):
            x = sorted(points)
            axis.plot(np.asarray(x) / 1e6, [statistics.mean(points[value]) for value in x], label=architecture)
        axis.set_title(cohort.title())
        axis.set_xlabel("Training tokens (millions)")
        axis.set_ylabel("Validation cross-entropy")
        axis.grid(alpha=0.25)
        if grouped:
            axis.legend()
    figure.suptitle("Registered-token learning curves")
    path = figure_root / "learning_curves_by_corpus.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 7))
    for architecture in sorted({str(row["architecture"]) for row in observations}):
        subset = [
            row
            for row in observations
            if row["architecture"] == architecture
            and _finite(row.get("validation_loss"))
            and _finite(row.get("test_loss"))
        ]
        if subset:
            axis.scatter(
                [float(row["validation_loss"]) for row in subset],
                [float(row["test_loss"]) for row in subset],
                label=architecture,
                alpha=0.55,
                s=28,
            )
    limits = axis.get_xlim()
    lower = min(limits[0], axis.get_ylim()[0])
    upper = max(limits[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], color="#64748b", linestyle="--", linewidth=1, label="validation = test")
    axis.set_xlabel("Validation cross-entropy at 50M tokens")
    axis.set_ylabel("Held-out test cross-entropy at 50M tokens")
    axis.set_title("Validation-to-test generalization by seed")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    path = figure_root / "validation_test_generalization.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    mechanism_rows = result["mechanism"]["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for architecture in ("T-KAM-L", "T-KAM-ALT"):
        subset = [row for row in mechanism_rows if row["architecture"] == architecture]
        axes[0].scatter([architecture] * len(subset), [row["freeze_fraction"] for row in subset], alpha=0.7)
        axes[1].scatter([architecture] * len(subset), [max(float(row["post_freeze_geometry_drift"] or 0.0), 1e-16) for row in subset], alpha=0.7)
    axes[0].axhspan(0.79, 0.81, color="#bbf7d0", alpha=0.5)
    axes[0].set_ylabel("Freeze tokens / total tokens")
    axes[0].set_title("Freeze timing by seed")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Post-freeze geometry drift")
    axes[1].set_title("Frozen-geometry integrity")
    path = figure_root / "learned_memory_lifecycle.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 6))
    for architecture in sorted({str(row["architecture"]) for row in observations}):
        subset = [row for row in observations if row["architecture"] == architecture and _finite(row.get("test_loss")) and _finite(row.get("estimated_training_flops"))]
        if subset:
            axis.scatter(
                [statistics.mean(float(row["estimated_training_flops"]) for row in subset)],
                [statistics.mean(float(row["test_loss"]) for row in subset)],
                label=architecture,
                s=55,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Estimated training FLOPs")
    axis.set_ylabel("Held-out test loss")
    axis.set_title("Matched-token resource–quality outcomes")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    path = figure_root / "resource_quality.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))
    return paths


def final_aggregate(run_root: Path, report_root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or run_root / "manifest.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = _load_rows(run_root)
    result = evaluate_confirmation(rows, manifest_rows)
    report_root.mkdir(parents=True, exist_ok=True)
    exports = {
        "confirmation_seed_metrics": _write_parquet(run_root / "confirmation_seed_metrics.parquet", result["observations"]),
        "confirmation_comparisons": _write_parquet(run_root / "confirmation_comparisons.parquet", [result["primary"], result["replication"], *result["secondary"]]),
        "mechanism_audit": _write_parquet(run_root / "mechanism_audit.parquet", result["mechanism"]["rows"]),
    }
    figures = _build_figures(result, report_root)
    summary = {key: value for key, value in result.items() if key != "observations"}
    summary["exports"] = exports
    summary["figures"] = figures
    (run_root / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    report = [
        "# Phase 6 Confirmatory Evidence Report",
        "",
        f"**Decision:** `{summary['decision']}`",
        "",
        summary["rationale"],
        "",
        "## Locked design",
        "",
        "- Primary: 30 fresh paired T-KAM-F/T-WIDE seeds on TinyStories, held-out test cross-entropy at 50M tokens.",
        "- Practical superiority: upper bootstrap 95% bound for the paired log-loss ratio must be at or below log(0.98), with paired randomization p ≤ 0.05.",
        "- Independent replication: 24 fresh paired seeds on Tiny Shakespeare, favorable 95% interval and p ≤ 0.05.",
        "- Secondary controls: 12 paired seeds versus T0 and T-PKM with Holm correction.",
        "- Learned-memory audit: eight seeds each for T-KAM-L and T-KAM-ALT; all lifecycle invariants must pass.",
        "- No optional stopping or post-hoc seed extension; only infrastructure failures may be rerun with the same immutable row.",
        "",
        "## Primary and replication",
        "",
        "| Comparison | n | Relative change | 95% CI | p | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record, passed in ((summary["primary"], summary["primary_pass"]), (summary["replication"], summary["replication_pass"])):
        if record.get("paired_seeds"):
            report.append(
                f"| {record['comparison']} | {record['paired_seeds']} | {100 * record['geometric_relative_change']:.2f}% | "
                f"[{100 * record['ci_low_relative_change']:.2f}%, {100 * record['ci_high_relative_change']:.2f}%] | "
                f"{record['paired_randomization_p']:.4g} | {passed} |"
            )
    report.extend(
        [
            "",
            "## Validity and mechanism",
            "",
            f"- Guardrails passed: `{summary['guardrails']['passed']}`",
            f"- Learned-memory lifecycle passed: `{summary['learned_memory_lifecycle_pass']}`",
            f"- Mechanism rows passed: {summary['mechanism']['passed_rows']}/{summary['mechanism']['expected_rows']}",
            "",
            "The fixed-key promotion decision and learned-memory lifecycle verdict are separate. A fixed-key result must not be described as evidence that learned geometry helps.",
            "",
        ]
    )
    (report_root / "CONFIRMATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (report_root / "README.md").write_text(
        "# Phase 6 confirmation v2\n\nStart with `CONFIRMATION_REPORT.md`, then inspect `final_summary.json`, Parquet exports, and `figures/`.\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["evaluate_confirmation", "final_aggregate", "paired_log_ratio"]
