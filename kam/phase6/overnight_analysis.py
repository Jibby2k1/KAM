"""Gates, aggregation, and reports for the Phase 6 overnight campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .overnight_manifest import (
    PREFLIGHT_ROWS,
    WAVE1_ROWS,
    WAVE2_ROWS,
    WAVE3_ROWS,
    build_wave2_rows,
    build_wave3_rows,
    read_jsonl,
    write_manifest,
)
from .stats import bootstrap_ci, equivalence_test, exact_permutation_test, holm_adjust, paired_effect


FINAL_OUTCOMES = (
    "PROMOTE_SPARSE_KAM_MEMORY",
    "PROMOTE_FIXED_KEY_FAST_ALGEBRA",
    "PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY",
    "PROMOTE_CONVENTIONAL_MEMORY_BASELINE",
    "PROMOTE_WIDENED_TRANSFORMER",
    "RETAIN_AS_DIAGNOSTIC_ONLY",
    "STOP_KAM_SPECIFIC_DIRECTION",
)


def _json_rows(root: Path, wave: str) -> list[dict[str, Any]]:
    directory = root / "rows" / wave
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if key == "metrics":
            continue
        flat[key] = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
    for key, value in row.get("metrics", {}).items():
        flat[f"metric_{key}"] = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
    return flat


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in materialized),
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: Iterable[dict[str, Any]], *, require: bool = True) -> dict[str, Any]:
    materialized = [_flatten(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path.with_suffix(".jsonl"), materialized)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(materialized) if materialized else pa.table({"empty": pa.array([], type=pa.string())})
        pq.write_table(table, path)
        return {"path": str(path), "rows": len(materialized), "engine": "pyarrow"}
    except Exception as exc:  # noqa: BLE001 - local environment may omit parquet
        if require:
            raise RuntimeError(f"Parquet export failed for {path}: {type(exc).__name__}: {exc}") from exc
        return {"path": str(path.with_suffix(".jsonl")), "rows": len(materialized), "engine": "jsonl_fallback"}


def _metric(row: dict[str, Any], name: str, default: float = math.inf) -> float:
    value = row.get("metrics", {}).get(name, row.get(f"metric_{name}", row.get(name, default)))
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _gate(rows: list[dict[str, Any]], expected: int, *, wave: str) -> dict[str, Any]:
    ids = [str(row.get("row_id")) for row in rows]
    failures = [str(row.get("row_id")) for row in rows if row.get("status") != "pass"]
    nonfinite = [str(row.get("row_id")) for row in rows if not _finite(row.get("metrics", {}))]
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    smoke = [str(row.get("row_id")) for row in rows if bool(row.get("metrics", {}).get("smoke_override"))]
    result = {
        "wave": wave,
        "pass": len(rows) == expected and not failures and not nonfinite and not duplicates and not smoke,
        "expected_rows": expected,
        "observed_rows": len(rows),
        "failure_row_ids": failures,
        "nonfinite_row_ids": nonfinite,
        "duplicate_row_ids": duplicates,
        "smoke_row_ids": smoke,
        "status_counts": dict(Counter(str(row.get("status", "missing")) for row in rows)),
    }
    return result


def preflight_gate(run_root: Path) -> dict[str, Any]:
    rows = _json_rows(run_root, "preflight")
    result = _gate(rows, PREFLIGHT_ROWS, wave="preflight")
    wrong_gpu = [
        str(row.get("row_id"))
        for row in rows
        if "L4" not in str(row.get("metadata", {}).get("gpu_name", "")).upper()
    ]
    data_failures: list[str] = []
    for row in rows:
        metrics = row.get("metrics", {})
        if row.get("lane") == "language" and (
            not metrics.get("dataset_sha256")
            or not metrics.get("tokenizer_sha256")
            or metrics.get("split_overlap") is not False
        ):
            data_failures.append(str(row.get("row_id")))
        if row.get("lane") == "dynamics" and (
            _metric(row, "finite_fraction", 0.0) != 1.0
            or _metric(row, "target_variance", 0.0) <= 1e-8
            or _metric(row, "clip_boundary_fraction", 1.0) >= 0.05
            or not metrics.get("nonconstant_stream")
        ):
            data_failures.append(str(row.get("row_id")))
    result["wrong_gpu_row_ids"] = wrong_gpu
    result["data_quality_failure_row_ids"] = sorted(set(data_failures))
    result["pass"] = bool(result["pass"] and not wrong_gpu and not data_failures)
    rates: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics", {})
        unit = "tokens" if row.get("lane") == "language" else "samples"
        rate = metrics.get(f"{unit}_per_second")
        if isinstance(rate, (int, float)) and float(rate) > 0:
            rates[f"{row.get('lane')}:{row.get('architecture')}"].append(float(rate))
            rates[f"{unit}:default"].append(float(rate))
    result["rates"] = {key: statistics.median(values) for key, values in rates.items()}
    destination = run_root / "calibration.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise RuntimeError(f"preflight gate failed; see {destination}")
    return result


def _pareto(rows: list[dict[str, Any]], quality: str = "validation_loss") -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("status") == "pass" and math.isfinite(_metric(row, quality))]
    frontier: list[dict[str, Any]] = []
    for candidate in valid:
        objectives = (
            _metric(candidate, quality),
            _metric(candidate, "estimated_active_flops_per_token", _metric(candidate, "active_parameters_per_token")),
            _metric(candidate, "peak_vram_bytes"),
            _metric(candidate, "wall_seconds", _metric(candidate, "row_wall_seconds")),
        )
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            comparison = (
                _metric(other, quality),
                _metric(other, "estimated_active_flops_per_token", _metric(other, "active_parameters_per_token")),
                _metric(other, "peak_vram_bytes"),
                _metric(other, "wall_seconds", _metric(other, "row_wall_seconds")),
            )
            if all(left <= right for left, right in zip(comparison, objectives)) and any(
                left < right for left, right in zip(comparison, objectives)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def stage1_frontier(source: Path, run_root: Path, report_root: Path) -> dict[str, Any]:
    rows = read_jsonl(source)
    finite = [row for row in rows if row.get("status") == "pass" and _finite(row.get("metrics", {}))]
    if len(rows) < 3000:
        raise RuntimeError(f"completed Stage 1 frontier requires 3000 rows, found {len(rows)}")
    pareto = _pareto(finite, quality="loss")
    root = run_root
    _write_parquet(root / "stage1_frontier.parquet", finite)
    _write_parquet(root / "stage1_pareto.parquet", pareto)
    families = {
        "fixed": {"T-KAM-F"},
        "learned": {"T-KAM-L"},
        "alt": {"T-KAM-ALT"},
        "vp": {"T-KAM-VP"},
    }
    limits = {"fixed": 2, "learned": 2, "alt": 1, "vp": 1}
    selected: list[dict[str, Any]] = []
    for family, architectures in families.items():
        candidates = [
            row
            for row in pareto
            if str(row.get("architecture")) in architectures
            and _metric(row, "recall_at_k_against_exact", 1.0) >= 0.95
            and (family not in {"alt", "vp"} or _metric(row, "alternating_geometry_steps", _metric(row, "geometry_update_steps", 1.0)) > 0)
        ]
        selected.extend(sorted(candidates, key=lambda row: _metric(row, "loss"))[: limits[family]])
    selection = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "finite_rows": len(finite),
        "pareto_rows": len(pareto),
        "selected_row_ids": [row.get("row_id") for row in selected],
        "selected": selected,
    }
    (root / "stage1_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n")
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "STAGE1_FRONTIER_REANALYSIS.md").write_text(
        "# Stage 1 Frontier Reanalysis\n\n"
        f"- Source rows: {len(rows)}\n"
        f"- Finite valid rows: {len(finite)}\n"
        f"- Pareto rows: {len(pareto)}\n"
        f"- Selected configurations: {len(selected)} (maximum 6)\n\n"
        "The CPU reanalysis filters failed/nonfinite identities, router recall below 0.95, "
        "and ALT/VP rows without geometry updates before Pareto selection. The full selected "
        "records are in `results/phase6/overnight/stage1_selection.json`.\n",
        encoding="utf-8",
    )
    return selection


def aggregate_wave(run_root: Path, wave: str, *, report_root: Path) -> dict[str, Any]:
    expected = {"wave1": WAVE1_ROWS, "wave2": WAVE2_ROWS}[wave]
    rows = _json_rows(run_root, wave)
    gate = _gate(rows, expected, wave=wave)
    pareto = _pareto(rows)
    _write_parquet(run_root / f"{wave}_metrics.parquet", rows)
    _write_parquet(run_root / f"{wave}_pareto.parquet", pareto)
    (run_root / f"{wave}_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["pass"]:
        raise RuntimeError(f"{wave} gate failed")
    if wave == "wave1":
        manifest = write_manifest(build_wave2_rows(rows), run_root / "manifests" / "wave2.jsonl")
    else:
        manifest = write_manifest(build_wave3_rows(rows), run_root / "manifests" / "wave3.jsonl")
    gate["next_manifest"] = manifest
    (run_root / f"{wave}_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate


def validate_manifest(path: Path, expected: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [row["row_id"] for row in rows]
    result = {
        "path": str(path),
        "rows": len(rows),
        "expected": expected,
        "unique": len(ids) == len(set(ids)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if len(rows) != expected or not result["unique"]:
        raise RuntimeError(f"invalid generated manifest: {result}")
    return result


def _group_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _metric(row, metric)
        if row.get("status") == "pass" and math.isfinite(value):
            groups[str(row.get("architecture"))].append(value)
    return groups


def _paired_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in ("validation_loss", "heldout_nmse", "late_post_transition_loss"):
        groups = _group_metric(rows, metric)
        if not groups:
            continue
        comparator = "T-WIDE" if "T-WIDE" in groups else "T0"
        for architecture, values in groups.items():
            if architecture == comparator:
                continue
            baseline = groups.get(comparator, [])
            count = min(len(values), len(baseline))
            if count < 2:
                continue
            left, right = values[:count], baseline[:count]
            differences = [a - b for a, b in zip(left, right)]
            effect = paired_effect(right, left)
            ci = bootstrap_ci(differences)
            permutation = exact_permutation_test(right, left)
            output.append(
                {
                    "metric": metric,
                    "architecture": architecture,
                    "comparator": comparator,
                    "paired_seeds": count,
                    "mean_difference": statistics.mean(differences),
                    "relative_difference": statistics.mean(differences) / max(abs(statistics.mean(right)), 1e-12),
                    "bootstrap_ci_low": ci[0],
                    "bootstrap_ci_high": ci[1],
                    "standardized_paired_effect": effect["effect_size_dz"],
                    "exact_paired_permutation_p": permutation["p_value"],
                    "holm_adjusted_p": permutation["p_value"],
                    "equivalent": equivalence_test(
                        right,
                        left,
                        margin=(0.02 if metric == "validation_loss" else 0.05) * max(abs(statistics.mean(right)), 1e-12),
                    ).get("equivalent"),
                }
            )
    for metric in {str(row["metric"]) for row in output}:
        family = {str(index): float(row["exact_paired_permutation_p"]) for index, row in enumerate(output) if row["metric"] == metric}
        adjusted = holm_adjust(family)
        for index, row in enumerate(output):
            if row["metric"] == metric:
                row["holm_adjusted_p"] = adjusted[str(index)]
    return output


def _decision(rows: list[dict[str, Any]]) -> tuple[str, str]:
    language = _group_metric([row for row in rows if "language" in str(row.get("lane"))], "validation_loss")
    adaptation = _group_metric([row for row in rows if row.get("lane") == "adaptation"], "late_post_transition_loss")
    means = {name: statistics.mean(values) for name, values in language.items() if values}
    kam = {name: value for name, value in means.items() if name.startswith("T-KAM")}
    conventional = {name: value for name, value in means.items() if name in {"T-MEMTOK", "T-MOE", "T-PKM"}}
    if kam and "T-WIDE" in means:
        best_kam = min(kam, key=kam.get)
        if kam[best_kam] <= 0.98 * means["T-WIDE"]:
            outcome = "PROMOTE_FIXED_KEY_FAST_ALGEBRA" if best_kam == "T-KAM-F" else "PROMOTE_SPARSE_KAM_MEMORY"
            return outcome, f"{best_kam} beat T-WIDE by at least the registered 2% language threshold."
    adaptation_means = {name: statistics.mean(values) for name, values in adaptation.items() if values}
    kam_adaptation = {name: value for name, value in adaptation_means.items() if name.startswith("T-KAM")}
    controls = {name: value for name, value in adaptation_means.items() if not name.startswith("T-KAM")}
    if kam_adaptation and controls and min(kam_adaptation.values()) <= 0.95 * min(controls.values()):
        return "PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY", "KAM improved late post-transition loss by at least 5%."
    if conventional and means and min(conventional.values()) == min(means.values()):
        return "PROMOTE_CONVENTIONAL_MEMORY_BASELINE", "A conventional memory baseline had the lowest replicated language loss."
    if "T-WIDE" in means and "T0" in means and means["T-WIDE"] <= 0.98 * means["T0"]:
        return "PROMOTE_WIDENED_TRANSFORMER", "T-WIDE beat T0 while no KAM cleared its promotion gate."
    if kam and "T-WIDE" in means and min(kam.values()) > 1.05 * means["T-WIDE"]:
        return "STOP_KAM_SPECIFIC_DIRECTION", "KAM remained more than 5% worse than T-WIDE without a compensating adaptation result."
    return "RETAIN_AS_DIAGNOSTIC_ONLY", "No architecture cleared a preregistered promotion or stop threshold."


def _report_header(title: str, rows: list[dict[str, Any]]) -> str:
    return (
        f"# {title}\n\n"
        f"Campaign rows analyzed: {len(rows)}. Status counts: "
        f"`{dict(Counter(str(row.get('status')) for row in rows))}`.\n\n"
    )


def _build_figures(rows: list[dict[str, Any]], report_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def finish(figure, name: str) -> None:
        path = figure_root / name
        figure.tight_layout()
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        created.append(str(path))

    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = 0
    for row in rows:
        if "language" not in str(row.get("lane")):
            continue
        for subrun in row.get("metrics", {}).get("subruns", []):
            history = subrun.get("loss_history", [])
            if not history:
                continue
            axis.plot(
                [point.get("tokens", point.get("step", index)) for index, point in enumerate(history)],
                [point.get("validation_loss", math.nan) for point in history],
                alpha=0.55,
                label=str(row.get("architecture")) if plotted < 12 else None,
            )
            plotted += 1
    axis.set(title="Language validation learning curves", xlabel="Training tokens", ylabel="Validation cross-entropy")
    if plotted:
        axis.legend(fontsize=7, ncol=3)
    else:
        axis.text(0.5, 0.5, "No completed language trajectories", ha="center", va="center")
    finish(figure, "language_learning_curves.png")

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    trace_count = 0
    for row in rows:
        if "dynamics" not in str(row.get("lane")):
            continue
        for subrun in row.get("metrics", {}).get("subruns", []):
            prediction = subrun.get("prediction_trace", [])
            truth = subrun.get("truth_trace", [])
            error = subrun.get("absolute_error_trace", [])
            if prediction and truth and trace_count < 4:
                label = f"{row.get('architecture')}:{subrun.get('task')}"
                axes[0].plot(truth, alpha=0.65, label=f"true {label}")
                axes[1].plot(prediction, alpha=0.65, label=f"pred {label}")
                axes[2].semilogy([max(float(value), 1e-8) for value in error], alpha=0.7, label=label)
                trace_count += 1
    axes[0].set_ylabel("True (normalized)")
    axes[1].set_ylabel("Prediction")
    axes[2].set_ylabel("|error| (log)")
    axes[2].set_xlabel("Held-out timestep")
    axes[0].set_title("Dynamics prediction, truth, and absolute error")
    if trace_count:
        for axis in axes:
            axis.legend(fontsize=6, ncol=2)
    else:
        axes[1].text(0.5, 0.5, "No completed dynamics traces", ha="center", va="center")
    finish(figure, "dynamics_prediction_true_error.png")

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    memory_count = 0
    for row in rows:
        if not str(row.get("architecture", "")).startswith("T-KAM"):
            continue
        for subrun in row.get("metrics", {}).get("subruns", []):
            history = subrun.get("loss_history", [])
            if not history:
                continue
            steps = [point.get("step", index) for index, point in enumerate(history)]
            label = f"{row.get('architecture')}:{row.get('wave')}"
            axes[0].plot(steps, [point.get("memory_gate_mean", math.nan) for point in history], alpha=0.65, label=label)
            axes[1].semilogy(
                steps,
                [max(float(point.get("memory_key_grad_norm", 0.0)), 1e-12) for point in history],
                alpha=0.65,
                label=label,
            )
            freeze = subrun.get("geometry_freeze_step")
            if isinstance(freeze, (int, float)):
                axes[0].axvline(freeze, color="black", alpha=0.1)
                axes[1].axvline(freeze, color="black", alpha=0.1)
            memory_count += 1
    axes[0].set(title="Memory adaptation then final-tuning freeze", ylabel="Memory gate scale")
    axes[1].set(xlabel="Optimizer step", ylabel="Key-gradient norm (log)")
    if memory_count:
        axes[0].legend(fontsize=6, ncol=3)
    else:
        axes[0].text(0.5, 0.5, "No completed KAM trajectories", ha="center", va="center")
    finish(figure, "memory_adaptation_freeze.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    for architecture in sorted({str(row.get("architecture")) for row in rows}):
        selected = [row for row in rows if str(row.get("architecture")) == architecture and math.isfinite(_metric(row, "validation_loss"))]
        if selected:
            axis.scatter(
                [_metric(row, "active_parameters_per_token") for row in selected],
                [_metric(row, "validation_loss") for row in selected],
                s=28,
                alpha=0.7,
                label=architecture,
            )
    axis.set(title="Quality versus active parameters per token", xlabel="Active parameters/token", ylabel="Validation loss")
    if axis.collections:
        axis.legend(fontsize=6, ncol=3)
        axis.set_xscale("log")
    else:
        axis.text(0.5, 0.5, "No completed resource-quality rows", ha="center", va="center")
    finish(figure, "resource_quality_pareto.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    adaptation = [row for row in rows if row.get("lane") == "adaptation"]
    labels = [str(row.get("architecture")) for row in adaptation]
    early = [_metric(row, "early_post_transition_loss", 0.0) for row in adaptation]
    late = [_metric(row, "late_post_transition_loss", 0.0) for row in adaptation]
    positions = list(range(len(labels)))
    if labels:
        axis.bar([value - 0.2 for value in positions], early, width=0.4, label="early")
        axis.bar([value + 0.2 for value in positions], late, width=0.4, label="late")
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No completed adaptation rows", ha="center", va="center")
    axis.set(title="Online adaptation recovery", ylabel="Post-transition loss")
    finish(figure, "adaptation_recovery.png")
    return created


def final_aggregate(run_root: Path, report_root: Path) -> dict[str, Any]:
    wave3 = _json_rows(run_root, "wave3")
    gate = _gate(wave3, WAVE3_ROWS, wave="wave3")
    (run_root / "wave3_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not gate["pass"]:
        raise RuntimeError("wave3 gate failed")
    rows = sum((_json_rows(run_root, wave) for wave in ("preflight", "wave1", "wave2", "wave3")), [])
    manifests = sum(
        (read_jsonl(run_root / "manifests" / f"{wave}.jsonl") for wave in ("preflight", "wave1", "wave2", "wave3")),
        [],
    )
    paired = _paired_statistics(rows)
    adaptation = [row for row in rows if row.get("lane") == "adaptation"]
    deletion: list[dict[str, Any]] = []
    for row in rows:
        for subrun in row.get("metrics", {}).get("subruns", []):
            for diagnostic in subrun.get("deletion_metrics", []):
                deletion.append({"row_id": row.get("row_id"), "architecture": row.get("architecture"), **diagnostic})
    resources = [
        {
            "row_id": row.get("row_id"),
            "wave": row.get("wave"),
            "architecture": row.get("architecture"),
            "gpu_name": row.get("metadata", {}).get("gpu_name"),
            "wall_seconds": _metric(row, "wall_seconds", row.get("row_wall_seconds", math.inf)),
            "peak_vram_bytes": _metric(row, "peak_vram_bytes", 0.0),
            "total_parameters": _metric(row, "total_parameters", 0.0),
            "active_parameters_per_token": _metric(row, "active_parameters_per_token", 0.0),
        }
        for row in rows
    ]
    failures = [row for row in rows if row.get("status") != "pass"]
    exports = {
        "run_manifest": _write_parquet(run_root / "run_manifest.parquet", manifests),
        "all_metrics": _write_parquet(run_root / "all_metrics.parquet", rows),
        "paired_seed_metrics": _write_parquet(run_root / "paired_seed_metrics.parquet", paired),
        "adaptation_metrics": _write_parquet(run_root / "adaptation_metrics.parquet", adaptation),
        "deletion_metrics": _write_parquet(run_root / "deletion_metrics.parquet", deletion),
        "resource_metrics": _write_parquet(run_root / "resource_metrics.parquet", resources),
        "failures": _write_parquet(run_root / "failures.parquet", failures),
    }
    outcome, rationale = _decision(rows)
    if outcome not in FINAL_OUTCOMES:
        raise AssertionError("decision outside registered outcome set")
    report_root.mkdir(parents=True, exist_ok=True)
    figures = _build_figures(rows, report_root)
    language_rows = [row for row in rows if "language" in str(row.get("lane"))]
    dynamics_rows = [row for row in rows if "dynamics" in str(row.get("lane"))]
    reports = {
        "OVERNIGHT_EXECUTION_REPORT.md": _report_header("Phase 6 Overnight Execution Report", rows)
        + f"Final outcome: `{outcome}`.\n\n{rationale}\n\nMachine-readable exports: `{json.dumps(exports, sort_keys=True)}`.\n",
        "OVERNIGHT_LANGUAGE_REPORT.md": _report_header("Phase 6 Overnight Language Report", language_rows)
        + "Primary metrics are held-out cross-entropy, perplexity, throughput, active compute, VRAM, and quality/GPU-hour. "
        "Use `all_metrics.parquet` and `paired_seed_metrics.parquet` for inferential detail.\n",
        "OVERNIGHT_DYNAMICS_REPORT.md": _report_header("Phase 6 Overnight Dynamics Report", dynamics_rows)
        + "Dynamics rows report held-out NMSE, validation trajectories, optimizer phase counts, conditioning, and data-quality checks.\n",
        "OVERNIGHT_OPTIMIZATION_REPORT.md": _report_header("Phase 6 Overnight Optimization Report", rows)
        + "ALT rows expose algebra/geometry step counts; VP rows freeze geometry under stop-gradient. "
        "Stage 1 Pareto selection is documented in `STAGE1_FRONTIER_REANALYSIS.md`.\n",
        "OVERNIGHT_ADAPTATION_REPORT.md": _report_header("Phase 6 Overnight Adaptation Report", adaptation)
        + "The inferential unit is the seed. Held-out schedules are aggregated within seed; early/late loss, excess loss, "
        "recovery/reacquisition time, update FLOPs, and state bytes are retained in the Parquet export.\n",
        "OVERNIGHT_DECISION_MEMO.md": "# Phase 6 Overnight Decision Memo\n\n"
        + f"## Decision\n\n`{outcome}`\n\n## Rationale\n\n{rationale}\n\n"
        + "This is the only registered final outcome emitted by the automated gate. A completed run is not itself evidence for KAM.\n",
        "OVERNIGHT_REPRODUCIBILITY.md": "# Phase 6 Overnight Reproducibility\n\n"
        + "Each row records commit/dirty state, manifest hash, architecture, dataset/tokenizer checksums, seeds, precision, "
        "GPU and framework versions, parameter/resource accounting, budgets, throughput, checkpoints, and failure category. "
        "The immutable row manifests and Slurm dependency graph are under `results/phase6/overnight/`.\n",
    }
    for name, body in reports.items():
        (report_root / name).write_text(body, encoding="utf-8")
    summary = {
        "gate": gate,
        "row_count": len(rows),
        "decision": outcome,
        "rationale": rationale,
        "exports": exports,
        "figures": figures,
    }
    (run_root / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight-gate", "final"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-root", required=True)
        if name == "final":
            child.add_argument("--report-root", required=True)
    frontier = subparsers.add_parser("stage1-frontier")
    frontier.add_argument("--source", required=True)
    frontier.add_argument("--run-root", required=True)
    frontier.add_argument("--report-root", required=True)
    aggregate = subparsers.add_parser("aggregate-wave")
    aggregate.add_argument("--wave", choices=("wave1", "wave2"), required=True)
    aggregate.add_argument("--run-root", required=True)
    aggregate.add_argument("--report-root", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--path", required=True)
    validate.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()
    run_root = Path(getattr(args, "run_root", "results/phase6/overnight"))
    if args.command == "preflight-gate":
        result = preflight_gate(run_root)
    elif args.command == "stage1-frontier":
        result = stage1_frontier(Path(args.source), run_root, Path(args.report_root))
    elif args.command == "aggregate-wave":
        result = aggregate_wave(run_root, args.wave, report_root=Path(args.report_root))
    elif args.command == "validate-manifest":
        result = validate_manifest(Path(args.path), args.expected)
    else:
        result = final_aggregate(run_root, Path(args.report_root))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
