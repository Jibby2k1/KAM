from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kam.stats import paired_bootstrap_ci, paired_permutation_pvalue

from .table import read_json, write_json, write_table


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def collect_metrics(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(run_root.rglob("metrics.json")):
        if metrics_path.parent.name == "status":
            continue
        metrics = _read_json(metrics_path)
        if not metrics:
            continue
        row_config = metrics.get("phase3_row", {})
        if not isinstance(row_config, dict):
            row_config = {}
        resolved = _read_json(metrics_path.parent / "resolved_config.json")
        run_config = resolved.get("run", {}) if isinstance(resolved.get("run"), dict) else {}
        validation = metrics.get("final_validation", metrics.get("best_validation", {}))
        if not isinstance(validation, dict):
            validation = {}
        descriptive = metrics.get("validation_descriptive_metrics", {})
        if not isinstance(descriptive, dict):
            descriptive = {}
        test = metrics.get("final_test", {})
        if not isinstance(test, dict):
            test = {}
        flat: dict[str, Any] = {
            "run_id": metrics.get("run_id", row_config.get("run_id", metrics_path.parent.name)),
            "task": metrics.get("task", row_config.get("task", run_config.get("task"))),
            "variant": metrics.get("variant", row_config.get("variant", run_config.get("variant"))),
            "scale": row_config.get("scale", "unknown"),
            "trial": row_config.get("trial", -1),
            "seed_slot": row_config.get("seed_slot", -1),
            "seed": metrics.get("seed", row_config.get("seed", run_config.get("seed"))),
            "status": metrics.get("status", "unknown"),
            "parameter_count": metrics.get("parameter_count"),
            "trainable_parameter_count": metrics.get("trainable_parameter_count"),
            "initial_trainable_parameter_count": metrics.get("initial_trainable_parameter_count"),
            "memory_bank_parameter_count": metrics.get("memory_bank_parameter_count"),
            "memory_protocol": metrics.get("memory_protocol", row_config.get("memory_protocol", run_config.get("memory_protocol", "joint"))),
            "memory_freeze_step": metrics.get("memory_freeze_step", row_config.get("memory_freeze_step")),
            "memory_trace_points": metrics.get("memory_trace_points", 0),
            "training_steps": metrics.get("training_steps"),
            "total_seconds": metrics.get("total_seconds"),
            "peak_memory_megabytes": metrics.get("peak_memory_megabytes"),
            "tokens_or_samples_per_second": metrics.get("tokens_or_samples_per_second"),
            "mse": validation.get("mse"),
            "nmse": validation.get("nmse"),
            "nrmse": validation.get("nrmse"),
            "mae": validation.get("mae"),
            "test_mse": test.get("mse"),
            "test_mae": test.get("mae"),
            "descriptive_rmse": descriptive.get("rmse"),
            "descriptive_bias": descriptive.get("bias"),
            "descriptive_p90_abs_error": descriptive.get("p90_abs_error"),
            "descriptive_p95_abs_error": descriptive.get("p95_abs_error"),
            "descriptive_r2": descriptive.get("r2"),
            "descriptive_correlation": descriptive.get("correlation"),
            "run_path": str(metrics_path.parent),
            "memory_trace_path": str(metrics_path.parent / "memory_training_trace.csv"),
            "memory_support_trace_path": str(metrics_path.parent / "memory_support_trace.csv"),
        }
        phase3 = metrics.get("phase3", {})
        if isinstance(phase3, dict):
            prequential = phase3.get("prequential", {})
            if isinstance(prequential, dict):
                for key, value in prequential.items():
                    flat[f"prequential_{key}"] = value
            causal = phase3.get("causal_probe", {})
            if isinstance(causal, dict):
                for key, value in causal.items():
                    flat[f"causal_{key}"] = value
        metric_rows.append({key: _coerce(value) for key, value in flat.items()})
        deletion_rows.extend(_read_csv(metrics_path.parent / "deletion_curves.csv"))
        support = _read_json(metrics_path.parent / "support_diagnostics.json")
        if support:
            support["run_path"] = str(metrics_path.parent)
            stability_rows.append(support)
    return metric_rows, deletion_rows, stability_rows


def _pair_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("task"), row.get("scale"), row.get("trial"), row.get("seed"))
        grouped[key][str(row.get("variant"))] = row
    effects: list[dict[str, Any]] = []
    for key, variants in sorted(grouped.items(), key=lambda item: str(item[0])):
        baseline = variants.get("D0")
        candidate = variants.get("DD-b")
        if baseline is None or candidate is None:
            continue
        use_prequential = baseline.get("prequential_adaptive_late_post_transition_mse") is not None and candidate.get("prequential_adaptive_late_post_transition_mse") is not None
        if use_prequential:
            left = float(baseline["prequential_adaptive_late_post_transition_mse"])
            right = float(candidate["prequential_adaptive_late_post_transition_mse"])
            endpoint = "adaptive_late_post_transition_mse"
        else:
            if baseline.get("mse") is None or candidate.get("mse") is None:
                continue
            left = float(baseline["mse"])
            right = float(candidate["mse"])
            endpoint = "validation_mse"
        effect = left - right
        relative = effect / max(abs(left), 1e-12)
        effects.append({
            "task": key[0], "scale": key[1], "trial": key[2], "seed": key[3],
            "baseline": "D0", "candidate": "DD-b", "endpoint": endpoint,
            "baseline_loss": left, "candidate_loss": right,
            "absolute_improvement": effect, "relative_improvement": relative,
        })
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        by_group[(row["task"], row["scale"], row["endpoint"])].append(row)
    for group, group_rows in by_group.items():
        values = np.asarray([row["absolute_improvement"] for row in group_rows], dtype=float)
        relative_values = np.asarray([row["relative_improvement"] for row in group_rows], dtype=float)
        left = np.asarray([row["baseline_loss"] for row in group_rows], dtype=float)
        right = np.asarray([row["candidate_loss"] for row in group_rows], dtype=float)
        abs_ci = paired_bootstrap_ci(left, right, resamples=2000, seed=101)
        rel_ci = tuple(float(value) for value in np.quantile(
            relative_values[np.random.default_rng(101).integers(0, len(relative_values), size=(2000, len(relative_values)))].mean(axis=1), [0.025, 0.975]
        )) if len(relative_values) else (float("nan"), float("nan"))
        pvalue = paired_permutation_pvalue(left, right, permutations=2000, seed=101)
        for row in group_rows:
            row["group_n"] = len(group_rows)
            row["group_mean_absolute_improvement"] = float(values.mean())
            row["group_mean_relative_improvement"] = float(relative_values.mean())
            row["group_absolute_ci_low"] = abs_ci[0]
            row["group_absolute_ci_high"] = abs_ci[1]
            row["group_relative_ci_low"] = rel_ci[0]
            row["group_relative_ci_high"] = rel_ci[1]
            row["group_permutation_pvalue"] = pvalue
    return effects


def _protocol_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair staged DD-b against joint DD-b on validation and held-out test MSE."""

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("task"), row.get("scale"), row.get("trial"), row.get("seed"))
        grouped[key][str(row.get("variant"))] = row
    effects: list[dict[str, Any]] = []
    for key, variants in sorted(grouped.items(), key=lambda item: str(item[0])):
        joint = variants.get("DD-b")
        staged = variants.get("DD-b-staged")
        if joint is None or staged is None:
            continue
        for field, endpoint in (("mse", "validation_mse"), ("test_mse", "test_mse")):
            if joint.get(field) is None or staged.get(field) is None:
                continue
            joint_loss = float(joint[field])
            staged_loss = float(staged[field])
            effect = joint_loss - staged_loss
            effects.append({
                "task": key[0], "scale": key[1], "trial": key[2], "seed": key[3],
                "baseline": "DD-b", "candidate": "DD-b-staged", "endpoint": endpoint,
                "baseline_loss": joint_loss, "candidate_loss": staged_loss,
                "absolute_improvement": effect,
                "relative_improvement": effect / max(abs(joint_loss), 1e-12),
            })
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in effects:
        by_group[(row["task"], row["scale"], row["endpoint"])].append(row)
    for group_rows in by_group.values():
        values = np.asarray([row["absolute_improvement"] for row in group_rows], dtype=float)
        relative_values = np.asarray([row["relative_improvement"] for row in group_rows], dtype=float)
        left = np.asarray([row["baseline_loss"] for row in group_rows], dtype=float)
        right = np.asarray([row["candidate_loss"] for row in group_rows], dtype=float)
        abs_ci = paired_bootstrap_ci(left, right, resamples=2000, seed=313)
        draws = np.random.default_rng(313).integers(0, len(relative_values), size=(2000, len(relative_values)))
        rel_ci = tuple(float(value) for value in np.quantile(relative_values[draws].mean(axis=1), [0.025, 0.975]))
        pvalue = paired_permutation_pvalue(left, right, permutations=2000, seed=313)
        for row in group_rows:
            row["group_n"] = len(group_rows)
            row["group_mean_absolute_improvement"] = float(values.mean())
            row["group_mean_relative_improvement"] = float(relative_values.mean())
            row["group_absolute_ci_low"] = abs_ci[0]
            row["group_absolute_ci_high"] = abs_ci[1]
            row["group_relative_ci_low"] = rel_ci[0]
            row["group_relative_ci_high"] = rel_ci[1]
            row["group_permutation_pvalue"] = pvalue
    return effects


def _plot_protocol_effects(effects: list[dict[str, Any]], figure_root: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in effects:
        groups[(str(row["task"]), str(row["scale"]), str(row["endpoint"]))] = row
    if not groups:
        return
    labels = list(groups)
    means = [float(groups[label]["group_mean_relative_improvement"]) for label in labels]
    lows = [means[index] - float(groups[label]["group_relative_ci_low"]) for index, label in enumerate(labels)]
    highs = [float(groups[label]["group_relative_ci_high"]) - means[index] for index, label in enumerate(labels)]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.errorbar(x, means, yerr=[lows, highs], fmt="o", capsize=4, color="#4c78a8")
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{task}\n{scale}\n{endpoint}" for task, scale, endpoint in labels], rotation=20, ha="right")
    axis.set_ylabel("relative improvement of staged over joint")
    axis.set_title("Warmup-then-freeze versus joint DD-b")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "staged_vs_joint_effects.png", dpi=180)
    plt.close(fig)


def _figure_root(report_root: Path) -> Path:
    root = report_root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _representative_trace_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], tuple[dict[str, Any], Path]]:
    representatives: dict[tuple[str, str, str], tuple[dict[str, Any], Path]] = {}
    for row in rows:
        path = Path(str(row.get("run_path", ""))) / "memory_training_trace.csv"
        if not path.exists():
            continue
        key = (str(row.get("task")), str(row.get("scale")), str(row.get("variant")))
        current = representatives.get(key)
        preferred = row.get("trial") == 0 and row.get("seed_slot") == 0
        if current is None or preferred:
            representatives[key] = (row, path)
    return representatives


def _plot_memory_drift(rows: list[dict[str, Any]], figure_root: Path) -> None:
    representatives = _representative_trace_rows(rows)
    if not representatives:
        return
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for (task, scale, _variant), value in representatives.items():
        groups[(task, scale)].append(value)
    colors = {"DD-b": "#4c78a8", "DD-b-staged": "#f58518", "DR-b": "#59a14f", "DR-b-staged": "#e45756", "RF-b": "#7f7f7f"}
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, max(4.0, 3.2 * len(groups))), squeeze=False)
    for axis, ((task, scale), group) in zip(axes[:, 0], sorted(groups.items())):
        for row, path in sorted(group, key=lambda item: str(item[0].get("variant"))):
            data = _read_csv(path)
            if not data:
                continue
            x = np.asarray([float(item["step"]) for item in data])
            color = colors.get(str(row.get("variant")), "#4c78a8")
            for kind, linestyle in (("key", "-"), ("value", "--")):
                values = [item.get(f"memory_{kind}_relative_drift") for item in data]
                if not any(value not in {None, ""} for value in values):
                    continue
                y = np.asarray([float(value or 0.0) for value in values])
                axis.plot(x, y, color=color, linestyle=linestyle, linewidth=1.6, label=f"{row.get('variant')} {kind}s")
            freeze = row.get("memory_freeze_step")
            if freeze not in {None, "", "None"} and str(row.get("variant")) == "DD-b-staged":
                axis.axvline(float(freeze), color="#222222", linestyle=":", linewidth=1.0)
        axis.set_title(f"{task} / {scale}")
        axis.set_ylabel("relative bank drift")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("training step")
    fig.suptitle("Learned memory-bank adaptation and freeze boundary", y=0.995)
    fig.tight_layout()
    fig.savefig(figure_root / "memory_bank_drift.png", dpi=180)
    plt.close(fig)


def _plot_memory_support_adaptation(rows: list[dict[str, Any]], figure_root: Path) -> None:
    candidates = []
    for row, path in _representative_trace_rows(rows).values():
        support_path = Path(str(row.get("run_path", ""))) / "memory_support_trace.csv"
        if support_path.exists():
            candidates.append((row, support_path))
    if not candidates:
        return
    selected = next((item for item in candidates if str(item[0].get("variant")) == "DD-b-staged"), candidates[0])
    row, path = selected
    data = _read_csv(path)
    if not data:
        return
    steps = sorted({int(float(item["step"])) for item in data})
    supports = sorted({int(float(item["support"])) for item in data})
    step_index = {step: index for index, step in enumerate(steps)}
    support_index = {support: index for index, support in enumerate(supports)}
    matrices: dict[str, np.ndarray] = {
        "mean_attention": np.full((len(supports), len(steps)), np.nan),
        "key_relative_drift": np.full((len(supports), len(steps)), np.nan),
        "value_relative_drift": np.full((len(supports), len(steps)), np.nan),
    }
    for item in data:
        x = step_index[int(float(item["step"]))]
        y = support_index[int(float(item["support"]))]
        for field, matrix in matrices.items():
            value = item.get(field)
            if value not in {None, ""}:
                matrix[y, x] = float(value)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    titles = ["mean attention over supports", "key-bank relative drift by support", "value-bank relative drift by support"]
    cmaps = ["Blues", "Oranges", "Greens"]
    for axis, (field, matrix), title, cmap in zip(axes, matrices.items(), titles, cmaps):
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", origin="lower", interpolation="nearest", cmap=cmap)
        axis.set_ylabel("support")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, pad=0.01)
    freeze = row.get("memory_freeze_step")
    if freeze not in {None, "", "None"}:
        nearest = int(np.argmin(np.abs(np.asarray(steps, dtype=float) - float(freeze))))
        for axis in axes:
            axis.axvline(nearest, color="#222222", linestyle=":", linewidth=1.0)
    axes[-1].set_xticks(np.arange(len(steps)))
    axes[-1].set_xticklabels([str(step) for step in steps], rotation=45, ha="right")
    axes[-1].set_xlabel("training step; dotted line = memory freeze")
    fig.suptitle(f"Memory-support adaptation: {row.get('task')} / {row.get('variant')} / {row.get('scale')}", y=0.995)
    fig.tight_layout()
    fig.savefig(figure_root / "memory_support_adaptation.png", dpi=180)
    plt.close(fig)


def _plot_train_validation_test(rows: list[dict[str, Any]], figure_root: Path) -> None:
    representatives = _representative_trace_rows(rows)
    selected = next((value for key, value in representatives.items() if key[2] == "DD-b-staged"), None)
    if selected is None:
        return
    row, path = selected
    data = _read_csv(path)
    if not data:
        return
    x = np.asarray([float(item["step"]) for item in data])
    fig, axis = plt.subplots(figsize=(9, 5))
    colors = {"train_mse": "#4c78a8", "validation_mse": "#f58518", "test_mse": "#59a14f"}
    labels = {"train_mse": "train", "validation_mse": "validation", "test_mse": "held-out test"}
    for field in ("train_mse", "validation_mse", "test_mse"):
        values = [item.get(field) for item in data]
        if not any(value not in {None, ""} for value in values):
            continue
        axis.plot(x, np.maximum(np.asarray([float(value or 0.0) for value in values]), 1e-12), marker="o", linewidth=1.5, color=colors[field], label=labels[field])
    freeze = row.get("memory_freeze_step")
    if freeze not in {None, "", "None"}:
        axis.axvline(float(freeze), color="#222222", linestyle=":", linewidth=1.0, label="memory freeze")
    axis.set_yscale("log")
    axis.set_xlabel("training step")
    axis.set_ylabel("MSE (log scale)")
    axis.set_title(f"Train/validation/test through staged tuning: {row.get('task')} / {row.get('scale')}")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "memory_train_validation_test.png", dpi=180)
    plt.close(fig)


def _plot_learning_curves(rows: list[dict[str, Any]], figure_root: Path) -> None:
    plt.figure(figsize=(8, 5))
    plotted = False
    seen: set[tuple[str, str]] = set()
    for row in rows:
        path = Path(str(row.get("run_path", ""))) / "metrics.json"
        metrics = _read_json(path)
        history = metrics.get("history", [])
        if not history:
            continue
        key = (str(row.get("scale")), str(row.get("variant")))
        if key in seen:
            continue
        seen.add(key)
        xs = [point.get("step") for point in history if isinstance(point, dict) and point.get("validation", {}).get("mse") is not None]
        ys = [point.get("validation", {}).get("mse") for point in history if isinstance(point, dict) and point.get("validation", {}).get("mse") is not None]
        if xs and ys:
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"{key[0]} {key[1]}")
            plotted = True
    if plotted:
        plt.yscale("log")
        plt.xlabel("training step")
        plt.ylabel("validation MSE (log scale)")
        plt.title("Phase III learning curves")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(figure_root / "learning_curves.png", dpi=180)
    plt.close()


def _plot_prediction_error(rows: list[dict[str, Any]], figure_root: Path) -> None:
    selected = None
    for row in rows:
        path = Path(str(row.get("run_path", "")))
        candidate = path / "prequential_trace.csv"
        if candidate.exists():
            selected = (row, candidate, "prequential")
            break
        candidate = path / "validation_predictions.csv"
        if candidate.exists():
            selected = (row, candidate, "validation")
            break
    if selected is None:
        return
    row, path, kind = selected
    data = _read_csv(path)
    if not data:
        return
    x = np.arange(len(data))
    target = np.asarray([float(item.get("target", 0.0)) for item in data])
    prediction_key = "adaptive_prediction" if kind == "prequential" else "prediction"
    error_key = "adaptive_error" if kind == "prequential" else "error"
    prediction = np.asarray([float(item.get(prediction_key, 0.0)) for item in data])
    error = np.asarray([float(item.get(error_key, 0.0)) for item in data])
    limit = min(len(x), 2000)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x[:limit], target[:limit], label="true", linewidth=1.0)
    axes[0].plot(x[:limit], prediction[:limit], label="prediction", linewidth=1.0)
    axes[0].set_ylabel("target")
    axes[0].legend()
    axes[1].plot(x[:limit], error[:limit], color="tab:red", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_ylabel("signed error")
    axes[2].plot(x[:limit], np.maximum(np.abs(error[:limit]), 1e-10), color="tab:purple", linewidth=0.8)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("|error| (log)")
    axes[2].set_xlabel("evaluation index")
    fig.suptitle(f"{row.get('task')} / {row.get('variant')} / {row.get('scale')}")
    fig.tight_layout()
    fig.savefig(figure_root / "prediction_true_error.png", dpi=180)
    plt.close(fig)


def _plot_error_distribution(rows: list[dict[str, Any]], figure_root: Path) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        path = Path(str(row.get("run_path", "")))
        trace_path = path / "prequential_trace.csv"
        prediction_path = path / "validation_predictions.csv"
        data = _read_csv(trace_path if trace_path.exists() else prediction_path)
        if not data:
            continue
        key = str(row.get("variant"))
        error_key = "adaptive_abs_error" if trace_path.exists() else "abs_error"
        values[key].extend(abs(float(item.get(error_key, 0.0))) for item in data)
    if not values:
        return
    plt.figure(figsize=(8, 5))
    for key, errors in sorted(values.items()):
        transformed = np.log10(np.maximum(np.asarray(errors, dtype=float), 1e-10))
        plt.hist(transformed, bins=40, alpha=0.45, density=True, label=key)
    plt.xlabel("log10 absolute error")
    plt.ylabel("density")
    plt.title("Phase III error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "error_distribution_log.png", dpi=180)
    plt.close()


def _plot_effects(effects: list[dict[str, Any]], figure_root: Path) -> None:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in effects:
        groups[(str(row["task"]), str(row["scale"]))] = row
    if not groups:
        return
    labels = list(groups)
    means = [float(groups[label]["group_mean_relative_improvement"]) for label in labels]
    lows = [means[index] - float(groups[label]["group_relative_ci_low"]) for index, label in enumerate(labels)]
    highs = [float(groups[label]["group_relative_ci_high"]) - means[index] for index, label in enumerate(labels)]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    plt.errorbar(x, means, yerr=[lows, highs], fmt="o", capsize=4)
    plt.axhline(0.15, color="tab:green", linestyle="--", label="15% practical threshold")
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.xticks(x, [f"{task}\n{scale}" for task, scale in labels], rotation=20, ha="right")
    plt.ylabel("relative late-loss improvement")
    plt.title("D0 vs DD-b paired Phase III effect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "primary_effects.png", dpi=180)
    plt.close()


def _write_report(report_root: Path, rows: list[dict[str, Any]], effects: list[dict[str, Any]], protocol_effects: list[dict[str, Any]], deletion_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    lines = [
        "# Phase III Development Search Report",
        "",
        f"Completed rows: **{len(complete):,} / {len(rows):,}**.",
        "",
        "The report is descriptive until the preregistered gates are evaluated. Development rows and confirmatory rows are not pooled.",
        "",
        "## Metrics",
        "",
        "The machine-readable table includes MSE, NMSE, NRMSE, MAE, bias, p90/p95 absolute error, R², correlation, parameter count, wall time, peak VRAM, and optional prequential NLMS endpoints.",
        "",
        "## Figures",
        "",
        "- `figures/learning_curves.png` — validation learning curves on a log-MSE axis.",
        "- `figures/prediction_true_error.png` — true value, prediction, signed error, and log absolute error.",
        "- `figures/error_distribution_log.png` — log10 absolute-error distributions.",
        "- `figures/primary_effects.png` — paired relative improvements and bootstrap intervals.",
        "- `figures/memory_bank_drift.png` — key/value support-bank movement, with the staged freeze boundary.",
        "- `figures/memory_support_adaptation.png` — support attention and per-support key/value drift over training.",
        "- `figures/memory_train_validation_test.png` — train, validation, and held-out test MSE through the freeze/tuning transition.",
        "- `figures/staged_vs_joint_effects.png` — paired staged-versus-joint validation/test effects.",
        "",
        f"Causal deletion rows: **{len(deletion_rows):,}**; support-stability rows: **{len(stability_rows):,}**.",
        "",
    ]
    if effects:
        lines.extend(["## D0 versus DD-b paired effects", "", "| Task | Scale | Endpoint | Pairs | Mean relative improvement | 95% CI | p-value |", "|---|---:|---|---:|---:|---:|---:|"])
        seen: set[tuple[Any, ...]] = set()
        for row in effects:
            key = (row["task"], row["scale"], row["endpoint"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {row['task']} | {row['scale']} | {row['endpoint']} | {row['group_n']} | {row['group_mean_relative_improvement']:.3f} | [{row['group_relative_ci_low']:.3f}, {row['group_relative_ci_high']:.3f}] | {row['group_permutation_pvalue']:.4f} |")
        lines.append("")
    if protocol_effects:
        lines.extend(["## Staged versus joint DD-b", "", "Positive values favor warmup-then-freeze; negative values favor joint training.", "", "| Task | Scale | Endpoint | Pairs | Mean relative improvement | 95% CI | p-value |", "|---|---:|---|---:|---:|---:|---:|"])
        seen: set[tuple[Any, ...]] = set()
        for row in protocol_effects:
            key = (row["task"], row["scale"], row["endpoint"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {row['task']} | {row['scale']} | {row['endpoint']} | {row['group_n']} | {row['group_mean_relative_improvement']:.3f} | [{row['group_relative_ci_low']:.3f}, {row['group_relative_ci_high']:.3f}] | {row['group_permutation_pvalue']:.4f} |")
        lines.append("")
    (report_root / "DEVELOPMENT_SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate(run_root: str | Path, output_path: str | Path | None = None, report_root: str | Path = "reports/phase3") -> dict[str, Any]:
    run_root = Path(run_root)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows, deletion_rows, stability_rows = collect_metrics(run_root)
    effects = _pair_effects(rows)
    protocol_effects = _protocol_effects(rows)
    output_path = Path(output_path or run_root / "phase3_aggregate.json")
    summary = {
        "run_root": str(run_root),
        "row_count": len(rows),
        "complete_count": sum(row.get("status") == "complete" for row in rows),
        "failed_count": sum(row.get("status") == "failed" for row in rows),
        "effect_group_count": len({(row["task"], row["scale"], row["endpoint"]) for row in effects}),
        "protocol_effect_group_count": len({(row["task"], row["scale"], row["endpoint"]) for row in protocol_effects}),
        "deletion_row_count": len(deletion_rows),
        "stability_row_count": len(stability_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, summary | {"effects": effects, "protocol_effects": protocol_effects})
    write_table(run_root / "all_metrics.parquet", rows)
    write_table(run_root / "seed_level_primary_effects.parquet", effects)
    write_table(run_root / "memory_protocol_effects.parquet", protocol_effects)
    write_table(run_root / "deletion_curves.parquet", deletion_rows)
    write_table(run_root / "support_stability.parquet", stability_rows)
    figures = _figure_root(report_root)
    _plot_learning_curves(rows, figures)
    _plot_prediction_error(rows, figures)
    _plot_error_distribution(rows, figures)
    _plot_effects(effects, figures)
    _plot_memory_drift(rows, figures)
    _plot_memory_support_adaptation(rows, figures)
    _plot_train_validation_test(rows, figures)
    _plot_protocol_effects(protocol_effects, figures)
    _write_report(report_root, rows, effects, protocol_effects, deletion_rows, stability_rows)
    results_lines = [
        "# Phase III Results",
        "",
        f"This development-stage aggregation covers {summary['complete_count']:,} complete rows and {summary['failed_count']:,} failed rows.",
        "",
        "Results are not confirmatory unless the machine-readable scientific gate passes on locked new seeds. Inspect the paired seed-level table and the figures before interpreting any apparent winner.",
        "",
        "See `DEVELOPMENT_SEARCH_REPORT.md` and `figures/` for descriptive metrics and plots.",
        "",
    ]
    (report_root / "PHASE3_RESULTS.md").write_text("\n".join(results_lines), encoding="utf-8")
    return summary | {"effects": effects, "protocol_effects": protocol_effects}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 rows and build descriptive reports.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.output, args.report_root), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
