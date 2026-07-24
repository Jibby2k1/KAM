"""Aggregate Phase IV screen outputs and build an auditable report bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _value(payload: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, dict) or payload.get(key) is None:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return None


def _load(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metrics_path in sorted((run_root / "runs").glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        spec = metrics.get("phase4_row", {})
        final = metrics.get("final_validation", {})
        train = metrics.get("final_train_evaluation", {}) or {}
        test = metrics.get("final_test", {}) or {}
        best = metrics.get("best_validation", {}) or {}
        rows.append({
            "run_id": metrics.get("run_id"), "task": spec.get("task", metrics.get("task")), "condition": spec.get("condition"),
            "variant": spec.get("variant", metrics.get("variant")), "scale": spec.get("scale"), "seed": spec.get("seed", metrics.get("seed")),
            "status": metrics.get("status"), "parameter_count": metrics.get("parameter_count"), "trainable_parameter_count": metrics.get("trainable_parameter_count"),
            "training_steps": metrics.get("training_steps"), "training_samples": metrics.get("training_samples"), "total_seconds": metrics.get("total_seconds"),
            "memory_protocol": metrics.get("memory_protocol"), "memory_freeze_step": metrics.get("memory_freeze_step"), "memory_trace_points": metrics.get("memory_trace_points"),
            "train_mse": _value(train, "mse"), "validation_mse": _value(final, "mse"), "best_validation_mse": _value(best, "mse"), "test_mse": _value(test, "mse"),
            "validation_mae": _value(final, "mae"), "test_mae": _value(test, "mae"), "run_path": str(metrics_path.parent),
        })
    for failure_path in sorted((run_root / "runs").glob("*/failure.json")):
        try:
            failures.append(json.loads(failure_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            failures.append({"path": str(failure_path), "error": "invalid failure JSON"})
    return rows, failures


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["condition"], row["scale"], row["variant"])].append(row)
    d0: dict[tuple[Any, ...], float] = {}
    for key, group in groups.items():
        if key[-1] == "D0":
            values = [float(x["test_mse"]) for x in group if x["test_mse"] is not None]
            if values:
                d0[key[:-1]] = float(np.mean(values))
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        tests = [float(x["test_mse"]) for x in group if x["test_mse"] is not None]
        vals = [float(x["validation_mse"]) for x in group if x["validation_mse"] is not None]
        baseline = d0.get(key[:-1])
        output.append({
            "task": key[0], "condition": key[1], "scale": key[2], "variant": key[3], "n": len(group),
            "mean_validation_mse": float(np.mean(vals)) if vals else None, "sd_validation_mse": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "mean_test_mse": float(np.mean(tests)) if tests else None, "sd_test_mse": float(np.std(tests, ddof=1)) if len(tests) > 1 else None,
            "relative_test_improvement_vs_D0_percent": (100.0 * (baseline - float(np.mean(tests))) / baseline) if baseline and tests else None,
        })
    return output


def _figures(rows: list[dict[str, Any]], report_root: Path) -> list[str]:
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    colors = {"D0": "#555555", "DD-b": "#1565c0", "DD-b-staged": "#2e7d32", "RF-b": "#ef6c00"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    axes = axes[0]
    for axis, task in zip(axes, sorted({row["task"] for row in rows})):
        for variant in colors:
            histories = []
            for row in rows:
                if row["task"] != task or row["variant"] != variant:
                    continue
                path = Path(row["run_path"]) / "metrics.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                history = payload.get("history", [])
                histories.append([(x.get("step"), (x.get("validation") or {}).get("mse")) for x in history])
            for history in histories:
                history = [(x, y) for x, y in history if y is not None]
                if history:
                    axis.plot([x for x, _ in history], [y for _, y in history], color=colors[variant], alpha=0.25)
        axis.set_title(task.replace("_", " "))
        axis.set_xlabel("training step")
        axis.set_ylabel("validation MSE")
    fig.suptitle("Phase IV learning curves (individual paired runs)")
    fig.tight_layout()
    fig.savefig(figure_root / "learning_curves.png", dpi=180)
    plt.close(fig)
    created.append("figures/learning_curves.png")

    selected = [row for row in rows if row["variant"] in {"DD-b", "DD-b-staged"}]
    if selected:
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
        for row in selected:
            pred_path = Path(row["run_path"]) / "test_predictions.csv"
            if not pred_path.exists():
                continue
            with pred_path.open(newline="", encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))[:500]
            index = sorted({row["task"] for row in selected}).index(row["task"])
            col = 0 if row["variant"] == "DD-b" else 1
            axis = axes[index, col]
            target = np.array([float(x["target"]) for x in prediction_rows])
            prediction = np.array([float(x["prediction"]) for x in prediction_rows])
            error = prediction - target
            axis.plot(target, label="true", linewidth=1)
            axis.plot(prediction, label="prediction", linewidth=1)
            axis2 = axis.twinx()
            axis2.plot(np.maximum(np.abs(error), 1e-8), color="#c62828", alpha=0.35, linewidth=0.7)
            axis2.set_yscale("log")
            axis.set_title(f"{row['task']} / {row['variant']} / {row['condition']}")
            axis.set_ylabel("target / prediction")
            axis2.set_ylabel("|error| (log)")
            axis.legend(loc="upper left", fontsize=7)
        fig.suptitle("Held-out prediction, truth, and absolute error")
        fig.tight_layout()
        fig.savefig(figure_root / "prediction_true_error.png", dpi=180)
        plt.close(fig)
        created.append("figures/prediction_true_error.png")

    trace_rows = []
    for row in rows:
        trace_path = Path(row["run_path"]) / "memory_training_trace.csv"
        if trace_path.exists():
            with trace_path.open(newline="", encoding="utf-8") as handle:
                for item in csv.DictReader(handle):
                    item.update({"task": row["task"], "variant": row["variant"], "condition": row["condition"], "scale": row["scale"]})
                    trace_rows.append(item)
    if trace_rows:
        fig, axis = plt.subplots(figsize=(9, 5))
        for variant in ["DD-b", "DD-b-staged", "RF-b"]:
            values = [(float(x["step"]), float(x.get("memory_key_relative_drift") or 0.0)) for x in trace_rows if x["variant"] == variant]
            if values:
                grouped: dict[float, list[float]] = defaultdict(list)
                for step, value in values:
                    grouped[step].append(value)
                steps = sorted(grouped)
                axis.plot(steps, [np.mean(grouped[x]) for x in steps], label=variant, color=colors[variant])
        axis.set_xlabel("training step")
        axis.set_ylabel("relative key drift from initialization")
        axis.set_title("Learned support-bank movement")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_root / "memory_drift.png", dpi=180)
        plt.close(fig)
        created.append("figures/memory_drift.png")

    return created


def _write_reports(rows: list[dict[str, Any]], failures: list[dict[str, Any]], summary: list[dict[str, Any]], report_root: Path, expected: int | None, figures: list[str]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    complete = len(rows)
    failed = len(failures)
    headline = f"{complete} completed runs, {failed} failed runs"
    best = sorted((x for x in summary if x.get("relative_test_improvement_vs_D0_percent") is not None), key=lambda x: x["relative_test_improvement_vs_D0_percent"], reverse=True)
    best_lines = [f"- `{x['task']}` / `{x['condition']}` / `{x['scale']}` / `{x['variant']}`: {x['relative_test_improvement_vs_D0_percent']:.2f}% relative test improvement vs D0 (n={x['n']})." for x in best[:8]] or ["- No complete paired D0 comparisons are available yet."]
    figure_lines = [f"![{Path(x).stem.replace('_', ' ')}]({x})" for x in figures]
    main = [
        "# Phase IV Factorial Mechanism Screen", "", f"**Status:** {headline}.", "",
        "## Technical summary", "",
        "This is a bounded Stage B development screen, not a confirmatory result. It tests whether currently supported learned-memory controls change with a small set of reproducible recurrence, coefficient-separation, and delay-separation conditions. Positive improvement means lower held-out test MSE than the paired D0 control.", "",
        f"Expected manifest rows: `{expected if expected is not None else 'not recorded'}`; observed metric rows: `{complete}`; failed rows: `{failed}`.", "",
        "## Largest descriptive effects", "", *best_lines, "",
        "These are descriptive paired summaries with two seeds per cell. They do not establish a causal mechanism, generalize to untested factors, or justify a confirmatory decision.", "",
        "## Visual evidence", "", *figure_lines, "",
        "## Scope and metric definitions", "",
        "- `D0` is the no-persistent-memory baseline; `DD-b` is the jointly trained learned bank; `DD-b-staged` is the existing warmup-then-freeze proxy for `DD-KV75`; `RF-b` is a frozen random-bank control.",
        "- Test checkpoints are selected using validation loss only. Test MSE is reported after selection and is not used for training or checkpoint choice.",
        "- The `recurring` and `separated` conditions are task-specific controls defined in `kam/phase4/manifest.py`; they are not yet the full factor library specified for Phase IV.", "",
        "## Limitations and next steps", "",
        "1. Add the missing controlled generators and freeze policies from the authoritative Phase IV brief, especially full-path freezing, drift-triggered freezing, noise-type controls, observability, and symbolic regimes.",
        "2. Expand the screen only after checking the paired effects and resource profile here.",
        "3. Promote no condition to confirmation without new seeds, held-out schedules/streams, registered endpoints, and paired uncertainty intervals.", "",
        "## Reproducibility", "", "- Manifest: `results/phase4/manifests/factorial_screen.jsonl`", "- Raw runs: `results/phase4/factorial_screen/runs/`", "- Aggregated metrics: `results/phase4/factorial_screen/all_metrics.csv` and `summary.csv`", "- Figures: `reports/phase4/figures/`", "- Execution: `scripts/submit_phase4_hpg.sh --submit`", "",
    ]
    (report_root / "PHASE4_FACTORIAL_REPORT.md").write_text("\n".join(main) + "\n", encoding="utf-8")
    handoff = [
        "# ChatGPT handoff: KAM Phase IV", "",
        "## Ask", "Advise on the next experiment stage using the attached repository artifacts. Treat this as a development screen, not confirmatory evidence.", "",
        "## Current status", f"- Expected rows: `{expected}`; complete: `{complete}`; failed: `{failed}`.", "- Primary report: `reports/phase4/PHASE4_FACTORIAL_REPORT.md`.", "- Aggregate table: `results/phase4/factorial_screen/all_metrics.csv`; grouped table: `results/phase4/factorial_screen/summary.csv`.", "",
        "## Design", "- Tasks: prototype switch, switching NARMA, switching Mackey–Glass.", "- Conditions: task-specific recurring versus separated controls.", "- Variants: D0, jointly trained DD-b, warmup-then-freeze DD-b-staged, and frozen random RF-b.", "- Scales: S and M; two paired seeds per cell; validation-selected checkpoints; held-out test metrics.", "",
        "## Interpretation guardrails", "- Relative improvement is `(D0 test MSE - candidate test MSE) / D0 test MSE`.", "- This screen does not test all Phase IV hypotheses and cannot establish stochasticity, coordinate mismatch, or causal support use.", "- Inspect `figures/learning_curves.png`, `figures/prediction_true_error.png`, and `figures/memory_drift.png` alongside the grouped table.", "",
        "## Questions for advice", "1. Which observed task × condition × variant effects merit Stage C freeze-policy search?", "2. Should the next screen prioritize noise/observability controls or full-path/drift-triggered freezing?", "3. What minimum paired seed/stream design would make the top mechanism claim credible?", "4. Which negative result would justify simplifying to generic adaptive readout?", "",
    ]
    (report_root / "PHASE4_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    memo = ["# Phase IV decision memo", "", f"Current decision: `RETAIN_AS_DIAGNOSTIC_ONLY` until the missing controlled factors and confirmatory design are run.", "", "The bounded screen is useful for prioritization only. No architecture is promoted from this result bundle. Revisit after Stage C and locked confirmation.", ""]
    (report_root / "PHASE4_DECISION_MEMO.md").write_text("\n".join(memo), encoding="utf-8")
    repro = ["# Phase IV reproducibility", "", "- Config: `configs/phase4/factorial_screen.yaml`", "- Manifest builder: `python -m kam.phase4.manifest --config configs/phase4/factorial_screen.yaml`", "- HPG submission: `scripts/submit_phase4_hpg.sh --submit`", "- Array runner: `python -m kam.phase4.run_array --manifest ... --array-index N --run-root ... --device auto --resume`", "- Aggregation: `python -m kam.phase4.aggregate --run-root results/phase4/factorial_screen --report-root reports/phase4`", "- Expected rows: `96` for the default config.", "- Paired unit: task × condition × scale × seed; variants share the seed and generated stream.", ""]
    (report_root / "PHASE4_REPRODUCIBILITY.md").write_text("\n".join(repro), encoding="utf-8")
    writeup = ["# Phase IV repository write-up", "", "Phase IV adds a bounded data-regime mechanism screen to the KAM experiments. It compares no memory, jointly learned memory, warmup-then-freeze memory, and a frozen random-bank control across prototype-switch, switching NARMA, and switching Mackey–Glass tasks. The design is manifest-driven, resumable on HiPerGator, and reports validation-selected held-out test metrics plus learning, prediction/error, and memory-drift figures.", "", "This first screen is intentionally a development instrument. It does not claim that a mechanism is established; the next step is to use its paired effects to prioritize the full freeze-policy, noise/observability, and confirmatory stages described in the Phase IV brief.", ""]
    (report_root / "PHASE4_REPOSITORY_WRITEUP.md").write_text("\n".join(writeup), encoding="utf-8")


def aggregate(run_root: Path, report_root: Path, expected: int | None = None) -> dict[str, Any]:
    rows, failures = _load(run_root)
    summary = _summary(rows)
    _write_csv(run_root / "all_metrics.csv", rows)
    _write_csv(run_root / "summary.csv", summary)
    (run_root / "all_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
    figures = _figures(rows, report_root) if rows else []
    _write_reports(rows, failures, summary, report_root, expected, figures)
    payload = {"expected": expected, "complete": len(rows), "failed": len(failures), "figures": figures, "report_root": str(report_root)}
    (run_root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase IV run artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/phase4"))
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.report_root, args.expected), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
