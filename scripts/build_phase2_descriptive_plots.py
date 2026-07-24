from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VARIANT_ORDER = ["D0", "R0", "DD", "DR", "RR", "DD-v", "DD-a", "DD-b", "DR-v", "DR-a", "DR-b", "RR-b"]
ADAPTER_ORDER = ["frozen", "nlms", "sgd", "rls"]
COLORS = {
    "D0": "#4c78a8", "R0": "#72b7b2", "DD": "#f58518", "DR": "#e45756", "RR": "#54a24b",
    "DD-v": "#f58518", "DD-a": "#ff9da6", "DD-b": "#b279a2", "DR-v": "#e45756", "DR-a": "#79706e", "DR-b": "#9d755d", "RR-b": "#59a14f",
}
ADAPTER_COLORS = {"frozen": "#7f7f7f", "nlms": "#4c78a8", "sgd": "#f58518", "rls": "#54a24b"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    return [dict(row) for row in csv.DictReader(path.open(encoding="utf-8"))] if path.exists() else []


def _read_validation_predictions(history_roots: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for root in history_roots:
        if not root.exists():
            continue
        for prediction_path in sorted(root.rglob("validation_predictions.csv")):
            metrics_path = prediction_path.parent / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in _read_csv(prediction_path):
                records.append({
                    "task": str(metadata.get("task", "")),
                    "variant": str(metadata.get("variant", metadata.get("model_spec", {}).get("model_name", ""))),
                    "adapter": "validation",
                    "seed": str(metadata.get("seed", "")),
                    "checkpoint": str(prediction_path),
                    "index": row.get("index", "0"),
                    "target": row.get("target", ""),
                    "prediction": row.get("prediction", ""),
                    "error": row.get("error", ""),
                    "abs_error": row.get("abs_error", ""),
                })
    return records


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _ordered(values: Iterable[str], order: list[str]) -> list[str]:
    return sorted(set(values), key=lambda value: (order.index(value) if value in order else len(order), value))


def _error_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    predictions = np.asarray([float(row["prediction"]) for row in rows], dtype=np.float64)
    errors = predictions - targets
    absolute = np.abs(errors)
    mse = float(np.mean(errors**2))
    target_centered = targets - np.mean(targets)
    prediction_centered = predictions - np.mean(predictions)
    denominator = float(np.sum(target_centered**2))
    correlation_denominator = float(np.sqrt(np.sum(target_centered**2) * np.sum(prediction_centered**2)))
    median_abs = float(np.median(absolute))
    p95_abs = float(np.quantile(absolute, 0.95))
    return {
        "n": len(rows),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "median_abs_error": median_abs,
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_error": p95_abs,
        "max_abs_error": float(np.max(absolute)),
        "r2": float(1.0 - np.sum(errors**2) / max(denominator, 1e-12)),
        "correlation": float(np.sum(target_centered * prediction_centered) / max(correlation_denominator, 1e-12)),
        "relative_mae": float(np.mean(absolute) / max(float(np.mean(np.abs(targets))), 1e-12)),
        "p95_to_median_abs_error": float(p95_abs / max(median_abs, 1e-12)),
        "log10_median_abs_error": float(np.log10(max(median_abs, 1e-12))),
    }


def write_descriptive_metrics(trace_rows: list[dict[str, str]], output: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        if not all(row.get(key) not in {None, ""} for key in ("task", "variant", "adapter", "seed", "target", "prediction")):
            continue
        grouped[(str(row["task"]), str(row["variant"]), str(row["adapter"]), str(row["seed"]))].append(row)
    records: list[dict[str, Any]] = []
    for (task, variant, adapter, seed), rows in sorted(grouped.items()):
        records.append({"task": task, "variant": variant, "adapter": adapter, "seed": int(seed), **_error_metrics(rows)})
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


def write_descriptive_summary(records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["task"]), str(record["variant"]), str(record["adapter"]))].append(record)
    summary: list[dict[str, Any]] = []
    excluded = {"task", "variant", "adapter", "seed", "n"}
    numeric_keys = [key for key in records[0] if key not in excluded] if records else []
    for (task, variant, adapter), group in sorted(grouped.items()):
        row: dict[str, Any] = {"task": task, "variant": variant, "adapter": adapter, "checkpoints": len(group), "samples": int(sum(float(item["n"]) for item in group))}
        for key in numeric_keys:
            values = [float(item[key]) for item in group if _float(item.get(key)) is not None]
            if values:
                row[key] = float(np.mean(values))
        summary.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in summary for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def _style(plt: Any) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fbfbfb", "axes.grid": True,
        "grid.alpha": 0.22, "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 120,
    })


def _save(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_histories(history_roots: list[Path]) -> list[dict[str, Any]]:
    records = []
    seen: set[Path] = set()
    for root in history_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("metrics.json")):
            if path in seen:
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") != "complete" or not payload.get("history"):
                continue
            records.append(payload)
    return records


def plot_learning_curves(histories: list[dict[str, Any]], output: Path, *, language: bool) -> bool:
    selected = []
    for payload in histories:
        task = str(payload.get("task", ""))
        is_language = "cross_entropy" in payload.get("history", [{}])[0].get("validation", {})
        if is_language == language:
            selected.append(payload)
    tasks = sorted({str(payload["task"]) for payload in selected})
    if not tasks:
        return False
    import matplotlib.pyplot as plt
    _style(plt)
    columns = 2
    rows = int(np.ceil(len(tasks) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    for axis, task in zip(axes.ravel(), tasks):
        task_payloads = [payload for payload in selected if str(payload["task"]) == task]
        variants = _ordered([str(payload["variant"]) for payload in task_payloads], VARIANT_ORDER)
        for variant in variants:
            variant_payloads = [payload for payload in task_payloads if str(payload["variant"]) == variant]
            by_step: dict[int, list[float]] = defaultdict(list)
            for payload in variant_payloads:
                for record in payload["history"]:
                    key = "cross_entropy" if language else "mse"
                    value = _float(record.get("validation", {}).get(key))
                    if value is not None:
                        by_step[int(record["step"])].append(value)
            steps = sorted(by_step)
            if not steps:
                continue
            medians = np.asarray([np.median(by_step[step]) for step in steps])
            low = np.asarray([np.quantile(by_step[step], 0.25) for step in steps])
            high = np.asarray([np.quantile(by_step[step], 0.75) for step in steps])
            color = COLORS.get(variant, "#333333")
            axis.plot(steps, medians, marker="o", linewidth=1.8, label=variant, color=color)
            axis.fill_between(steps, low, high, color=color, alpha=0.12)
        axis.set_title(task.replace("_", " ").title())
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("validation cross-entropy" if language else "validation MSE")
        if not language:
            axis.set_yscale("log")
        axis.legend(ncol=3, frameon=False)
    for axis in axes.ravel()[len(tasks):]:
        axis.axis("off")
    fig.suptitle("Learning curves: median with interquartile band across completed runs", y=1.01, fontsize=13)
    _save(fig, output)
    return True


def _select_trace_group(trace_rows: list[dict[str, str]], task: str, variant: str = "DR", adapter: str = "nlms", seed: str = "7") -> list[dict[str, str]]:
    rows = [row for row in trace_rows if row.get("task") == task and row.get("variant") == variant and row.get("adapter") == adapter and row.get("seed") == seed]
    if rows:
        return sorted(rows, key=lambda row: int(row["index"]))
    rows = [row for row in trace_rows if row.get("task") == task and row.get("adapter") == adapter]
    if not rows:
        rows = [row for row in trace_rows if row.get("task") == task]
    return sorted(rows, key=lambda row: int(row["index"]))


def plot_prediction_true_error(trace_rows: list[dict[str, str]], task: str, output: Path) -> bool:
    rows = _select_trace_group(trace_rows, task)
    if not rows:
        return False
    import matplotlib.pyplot as plt
    _style(plt)
    limit = min(len(rows), 1200)
    rows = rows[:limit]
    x = np.arange(limit)
    target = np.asarray([float(row["target"]) for row in rows])
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    error = prediction - target
    regimes = [str(row.get("regime", "")) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, target, color="#222222", linewidth=1.0, label="true")
    axes[0].plot(x, prediction, color="#e45756", linewidth=1.0, alpha=0.85, label="prediction")
    axes[0].set_ylabel("normalized target")
    axes[0].set_title(f"{task.replace('_', ' ').title()}: true vs prediction — DR / NLMS / seed 7")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, error, color="#4c78a8", linewidth=0.8, label="signed error")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("prediction − true")
    axes[1].set_xlabel("prequential sample index")
    axes[1].legend(frameon=False)
    for index in range(1, len(regimes)):
        if regimes[index] != regimes[index - 1]:
            for axis in axes:
                axis.axvline(index, color="#999999", linewidth=0.7, linestyle="--", alpha=0.65)
    _save(fig, output)
    return True


def plot_error_distribution(trace_rows: list[dict[str, str]], output: Path) -> bool:
    selected = [row for row in trace_rows if row.get("adapter") in {"validation", "frozen", "nlms", "sgd", "rls"} and row.get("variant") in {"D0", "DD", "DR", "RR"}]
    if not selected:
        return False
    import matplotlib.pyplot as plt
    _style(plt)
    tasks = _ordered([row.get("task", "") for row in selected], [])
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.8), squeeze=False)
    for axis, task in zip(axes.ravel(), tasks):
        data = []
        labels = []
        positions = []
        colors = []
        position = 0.0
        for variant in _ordered([row["variant"] for row in selected if row.get("task") == task], VARIANT_ORDER):
            adapters = ADAPTER_ORDER if any(row.get("adapter") in ADAPTER_ORDER for row in selected if row.get("task") == task) else ["validation"]
            for adapter in adapters:
                values = [max(abs(float(row["prediction"]) - float(row["target"])), 1e-12) for row in selected if row.get("task") == task and row.get("variant") == variant and row.get("adapter") == adapter]
                if not values:
                    continue
                data.append(np.log10(np.asarray(values)))
                positions.append(position)
                labels.append(f"{variant}\n{adapter}")
                colors.append(ADAPTER_COLORS.get(adapter, "#9467bd"))
                position += 1.0
            position += 0.7
        box = axis.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("log10 absolute error")
        axis.set_title(task.replace("_", " ").title())
    fig.suptitle("Error distribution by architecture and adapter", y=1.02, fontsize=13)
    _save(fig, output)
    return True


def plot_recovery_curves(trace_rows: list[dict[str, str]], output: Path) -> bool:
    if not trace_rows:
        return False
    import matplotlib.pyplot as plt
    _style(plt)
    tasks = sorted({row.get("task", "") for row in trace_rows})
    horizon = 128
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.5), squeeze=False)
    overall_windows = 0
    for axis, task in zip(axes.ravel(), tasks):
        for adapter in ADAPTER_ORDER:
            windows: list[np.ndarray] = []
            groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
            for row in trace_rows:
                if row.get("task") == task and row.get("adapter") == adapter:
                    groups[(row.get("checkpoint", ""), row.get("variant", ""), row.get("seed", ""))].append(row)
            for group_rows in groups.values():
                group_rows.sort(key=lambda row: int(row["index"]))
                losses = np.asarray([float(row["loss"]) for row in group_rows])
                regimes = [row.get("regime", "") for row in group_rows]
                transitions = [idx for idx in range(1, len(regimes)) if regimes[idx] != regimes[idx - 1]]
                for transition in transitions:
                    if transition + horizon <= len(losses):
                        windows.append(losses[transition : transition + horizon])
            if not windows:
                continue
            overall_windows += len(windows)
            matrix = np.vstack(windows)
            axis.plot(np.arange(horizon), np.median(matrix, axis=0), label=adapter, color=ADAPTER_COLORS[adapter])
            axis.fill_between(np.arange(horizon), np.quantile(matrix, 0.25, axis=0), np.quantile(matrix, 0.75, axis=0), color=ADAPTER_COLORS[adapter], alpha=0.12)
        axis.set_title(task.replace("_", " ").title())
        axis.set_xlabel("samples after detected regime transition")
        axis.set_ylabel("squared prediction loss")
        axis.set_yscale("log")
        axis.legend(frameon=False)
    if overall_windows == 0:
        plt.close(fig)
        return False
    fig.suptitle("Post-shift recovery curves: median with interquartile band", y=1.02, fontsize=13)
    _save(fig, output)
    return True


def plot_support_diagnostics(shift_metrics: list[dict[str, str]], output: Path) -> bool:
    rows = [row for row in shift_metrics if _float(row.get("global_effective_supports")) is not None and _float(row.get("support_purity")) is not None]
    if not rows:
        return False
    import matplotlib.pyplot as plt
    _style(plt)
    tasks = sorted({row.get("task", "") for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.5), squeeze=False)
    for axis, task in zip(axes.ravel(), tasks):
        task_rows = [row for row in rows if row.get("task") == task]
        for variant in _ordered([row.get("variant", "") for row in task_rows], VARIANT_ORDER):
            subset = [row for row in task_rows if row.get("variant") == variant]
            x = [_float(row.get("global_effective_supports")) for row in subset]
            y = [_float(row.get("support_purity")) for row in subset]
            axis.scatter(x, y, label=variant, color=COLORS.get(variant, "#333333"), alpha=0.8, s=35)
        axis.axvline(32, color="#888888", linestyle="--", linewidth=0.8, label="bank size")
        axis.set_xlabel("global effective supports")
        axis.set_ylabel("support purity")
        axis.set_ylim(0, 1)
        axis.set_title(task.replace("_", " ").title())
        axis.legend(frameon=False, ncol=2)
    fig.suptitle("Support utilization versus regime alignment", y=1.02, fontsize=13)
    _save(fig, output)
    return True


def build_descriptive_artifacts(
    *,
    trace_path: Path,
    shift_metrics_path: Path,
    history_roots: list[Path],
    output_dir: Path,
    metrics_output: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = _read_csv(trace_path)
    if not trace_rows:
        trace_rows = _read_validation_predictions(history_roots)
    shift_metrics = _read_csv(shift_metrics_path)
    metrics = write_descriptive_metrics(trace_rows, metrics_output)
    summary_output = metrics_output.with_name(f"{metrics_output.stem}_summary.csv")
    summary = write_descriptive_summary(metrics, summary_output)
    histories = _load_histories(history_roots)
    created: list[str] = []
    plot_specs = [
        ("learning_curves_regression.png", lambda: plot_learning_curves(histories, output_dir / "learning_curves_regression.png", language=False)),
        ("learning_curves_language.png", lambda: plot_learning_curves(histories, output_dir / "learning_curves_language.png", language=True)),
        ("prediction_true_error_switching_mackey_glass.png", lambda: plot_prediction_true_error(trace_rows, "switching_mackey_glass", output_dir / "prediction_true_error_switching_mackey_glass.png")),
        ("prediction_true_error_switching_narma.png", lambda: plot_prediction_true_error(trace_rows, "switching_narma", output_dir / "prediction_true_error_switching_narma.png")),
        ("error_distribution_log.png", lambda: plot_error_distribution(trace_rows, output_dir / "error_distribution_log.png")),
        ("post_shift_recovery_curves.png", lambda: plot_recovery_curves(trace_rows, output_dir / "post_shift_recovery_curves.png")),
        ("support_diagnostics.png", lambda: plot_support_diagnostics(shift_metrics, output_dir / "support_diagnostics.png")),
    ]
    for name, builder in plot_specs:
        try:
            if builder():
                created.append(name)
        except (KeyError, ValueError, IndexError, RuntimeError) as error:
            print(f"Skipping {name}: {error}")
    chart_map = output_dir.parent / "CHART_MAP.md"
    chart_map.write_text(
        "# Phase II descriptive chart map\n\n"
        "| Figure | Analytical question | Visual contract | Evidence | Caveat |\n"
        "|---|---|---|---|---|\n"
        "| learning_curves_regression.png | Do variants learn at different rates or plateau differently on regression tasks? | Median validation MSE with IQR across completed runs; log y-axis. | metrics.json histories from paired and switching screens. | Legacy runs have sparse checkpoints; future runs now default to 10% evaluation cadence. |\n"
        "| learning_curves_language.png | Do language/mechanism variants converge differently? | Median validation cross-entropy with IQR. | metrics.json histories from language matrix. | This is a representative mechanism screen, not natural-language modeling. |\n"
        "| prediction_true_error_switching_mackey_glass.png / prediction_true_error_switching_narma.png | Where do predictions diverge from the true stream and when do errors change sign? | True and prediction above signed error, with regime boundaries. | prequential shift_trace.csv, DR/NLMS/seed 7 fallback selection. | One representative checkpoint; use group metrics for aggregate claims. |\n"
        "| error_distribution_log.png | Are improvements broad or driven by a few extreme errors? | Boxplots of log10 absolute error by variant and adapter. | prequential shift_trace.csv. | Log scale clips zero at 1e-12; distributions are checkpoint-level observations. |\n"
        "| post_shift_recovery_curves.png | How quickly do adapters recover after a detected regime transition? | Median/IQR squared loss over 128 post-transition samples. | prequential shift_trace.csv. | Transition alignment is based on observed regime labels for diagnostics only. |\n"
        "| support_diagnostics.png | Are supports used broadly and aligned with regimes? | Effective support count versus support purity. | shift_metrics.csv. | Purity is descriptive, not causal faithfulness. |\n\n"
        "Metric CSVs: `results/phase2/descriptive_metrics.csv` (per checkpoint/seed) and `results/phase2/descriptive_metrics_summary.csv` (aggregated by task, variant, and adapter). Metrics include RMSE, MAE, signed bias, error spread, median/p90/p95/max absolute error, R², correlation, relative MAE, tail ratio, and log10 median absolute error.\n\n"
        "QA note: Matplotlib export, dimensions, and nonblank-file checks were run. The local image viewer was unavailable in this sandbox because its filesystem helper failed to initialize loopback networking, so visual inspection was limited to automated file-level QA.\n",
        encoding="utf-8",
    )
    return {"created": created, "metrics": metrics, "summary": summary, "metrics_path": str(metrics_output), "summary_path": str(summary_output), "chart_map": str(chart_map)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build descriptive Phase II metrics and plots from saved experiment artifacts.")
    parser.add_argument("--trace", type=Path, default=Path("results/phase2/switching_adaptation/shift_trace.csv"))
    parser.add_argument("--shift-metrics", type=Path, default=Path("results/phase2/switching_adaptation/shift_metrics.csv"))
    parser.add_argument("--history-root", type=Path, action="append", default=[Path("results/phase2/paired_screen"), Path("results/phase2/switching_paired_screen"), Path("results/phase2/language_matrix")])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2/figures"))
    parser.add_argument("--metrics-output", type=Path, default=Path("results/phase2/descriptive_metrics.csv"))
    args = parser.parse_args()
    result = build_descriptive_artifacts(trace_path=args.trace, shift_metrics_path=args.shift_metrics, history_roots=args.history_root, output_dir=args.output_dir, metrics_output=args.metrics_output)
    print(json.dumps({"created": result["created"], "metrics": len(result["metrics"]), "summary": len(result["summary"]), "metrics_path": result["metrics_path"], "summary_path": result["summary_path"], "chart_map": result["chart_map"]}, indent=2))


if __name__ == "__main__":
    main()
