"""Aggregate Stage 1 Phase V mechanism-pilot artifacts."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(run_root: Path, report_root: Path, expected: int) -> dict[str, Any]:
    metric_paths = sorted((run_root / "runs").glob("*/metrics.json"))
    failure_paths = sorted((run_root / "runs").glob("*/failure.json"))
    metrics_list = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    rows = []
    histories: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for metrics, path in zip(metrics_list, metric_paths):
        spec = metrics.get("phase5_row", {})
        test = metrics.get("best_checkpoint_test", {}) or {}
        rows.append({
            "run_id": metrics.get("run_id"), "task": spec.get("task"), "variant": spec.get("variant"),
            "scale": spec.get("scale"), "seed_index": spec.get("seed_index"),
            "active_parameter_count": metrics.get("active_parameter_count"),
            "target_active_parameters": spec.get("target_active_parameters"),
            "active_capacity_match_error": metrics.get("active_capacity_match_error"),
            "padding_parameter_count": metrics.get("padding_parameter_count"),
            "route_feature_dim": metrics.get("route_feature_dim"),
            "test_mse": test.get("mse"), "test_nmse": test.get("nmse"), "test_nrmse": test.get("nrmse"),
            "test_mae": test.get("mae"), "test_p95_to_median_abs_error": test.get("p95_to_median_abs_error"),
            "total_seconds": metrics.get("total_seconds"), "fixed_component": metrics.get("fixed_component"),
            "variant_semantics": metrics.get("variant_semantics"), "run_path": str(path.parent),
        })
        for record in metrics.get("history", []):
            value = record.get("validation", {}).get("nmse")
            if value is not None:
                histories[(str(spec.get("scale")), str(spec.get("variant")), int(record.get("step", 0)))].append(float(value))
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (run_root / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scale"]), str(row["variant"]))].append(row)
    summaries = []
    for (scale, variant), group in sorted(groups.items()):
        nmse = [float(row["test_nmse"]) for row in group if row["test_nmse"] is not None]
        summaries.append({
            "scale": scale, "variant": variant, "n": len(nmse),
            "mean_test_nmse": _mean(nmse), "sd_test_nmse": _sd(nmse),
            "mean_test_nrmse": _mean([float(row["test_nrmse"]) for row in group if row["test_nrmse"] is not None]),
            "mean_active_parameter_count": _mean([float(row["active_parameter_count"]) for row in group]),
            "max_active_capacity_match_error": max(float(row["active_capacity_match_error"]) for row in group),
            "mean_seconds": _mean([float(row["total_seconds"]) for row in group]),
        })
    with (run_root / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["scale", "variant"])
        writer.writeheader()
        writer.writerows(summaries)
    checks = {
        "expected_rows_complete": len(metric_paths) == expected,
        "no_failure_artifacts": not failure_paths,
        "all_rows_passed_pilot_checks": bool(metrics_list) and all(all(metric.get("phase5_pilot_checks", {}).values()) for metric in metrics_list),
        "zero_padding": bool(metrics_list) and all(int(metric.get("padding_parameter_count", 0)) == 0 for metric in metrics_list),
        "active_capacity_match": bool(metrics_list) and all(float(metric.get("active_capacity_match_error", 99.0)) <= 0.01 for metric in metrics_list),
    }
    checks.update({"passed": all(checks.values()), "expected": expected, "completed": len(metric_paths), "failed": len(failure_paths)})
    (run_root / "pilot_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")

    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    variants = ["D0", "DD-L", "RF-KV", "RF-FULL", "KC-LV", "RFF"]
    scales = ["P250", "P1M"]
    fig, axis = plt.subplots(figsize=(12, 5))
    width = 0.38
    for offset, scale in enumerate(scales):
        values = [next((float(item["mean_test_nmse"]) for item in summaries if item["scale"] == scale and item["variant"] == variant), float("nan")) for variant in variants]
        axis.bar([index + (offset - 0.5) * width for index in range(len(variants))], values, width=width, label=scale)
    axis.set_xticks(range(len(variants)), variants)
    axis.set_ylabel("held-out global NMSE")
    axis.set_title("Phase V Stage 1 pilot: mean held-out NMSE")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "pilot_nmse_by_variant.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    for scale in scales:
        for variant in variants:
            points = sorted((step, _mean(values)) for (point_scale, point_variant, step), values in histories.items() if point_scale == scale and point_variant == variant)
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], label=f"{scale} {variant}")
    axis.set_xlabel("training step")
    axis.set_ylabel("validation global NMSE")
    axis.set_title("Phase V Stage 1 learning curves")
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "pilot_learning_curves.png", dpi=180)
    plt.close(fig)

    status = "PASSED" if checks["passed"] else "FAILED"
    report = [
        "# Phase V mechanism pilot report", "",
        f"Pilot status: {status}. Completed {checks['completed']} of {checks['expected']} rows with {checks['failed']} failure artifacts.", "",
        "## Design", "",
        "This Stage 1 pilot uses four controlled task labels, six primary controls (D0, DD-L, RF-KV, RF-FULL, KC-LV, RFF), two active-capacity-matched scales, three paired seeds, fixed projected route dimension 64 for KAM models, independent streams, global held-out NMSE, and validation-selected checkpoint reload.", "",
        "The pilot uses the specified iid-window protocol for signal and variance profiling. Ordered recurrence and adaptation are reserved for the next stage.", "",
        "## Summary", "",
    ]
    report.extend([f"- {item['scale']} / {item['variant']}: mean NMSE={item['mean_test_nmse']:.4g}, SD={item['sd_test_nmse']:.4g}, n={item['n']}, mean active parameters={item['mean_active_parameter_count']:.0f}" for item in summaries])
    report += ["", "## Interpretation guardrails", "", "These results are a mechanism-pilot screen, not a confirmatory decision. The controlled Mackey–Glass, NARMA, prototype, and symbolic labels currently share the controlled-stream execution path; task-specific generator implementations remain a follow-up requirement before final claims.", "", "Read pilot_summary.csv for paired seed-level summaries and pilot_learning_curves.png for convergence behavior. Compare variants within scale and task; do not rank pooled means as if task difficulty were identical.", ""]
    (report_root / "PHASE5_MECHANISM_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    handoff = [
        "# ChatGPT handoff: KAM Phase V Stage 1", "",
        "## Ask", "Review the Stage 1 mechanism-pilot results and advise which controls and factors should advance to the multi-fidelity factorial search.", "",
        "## Status", f"- Pilot: {status}; {checks['completed']}/{checks['expected']} rows completed.", "- Machine checks: results/phase5/pilot/pilot_checks.json.", "- Metrics: results/phase5/pilot/all_metrics.csv.", "- Summary: results/phase5/pilot/pilot_summary.csv.", "- Technical report: reports/phase5/PHASE5_MECHANISM_REPORT.md.", "",
        "## Questions", "1. Which variant differences are stable across tasks and active-capacity scales?", "2. Which controls should be eliminated before factorial search?", "3. Which return probability, separation, observability, noise, and ordered-protocol factors should be prioritized?", "4. Are the paired-seed and capacity-match checks sufficient for promotion to Stage 2?", "",
    ]
    (report_root / "PHASE5_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    writeup = [
        "# Phase V repository write-up", "",
        f"The Stage 1 mechanism pilot {status.lower()} with {checks['completed']}/{checks['expected']} completed HPG runs.", "",
        "It compares six primary feature controls across four controlled task labels, two active-capacity scales, and three paired seeds. The report includes held-out NMSE summaries, convergence curves, active-capacity checks, and a ChatGPT handoff for selecting the next factorial search.", "",
        "The pilot is intended to identify signal direction and unusable controls. It is not a final scientific promotion decision.",
    ]
    (report_root / "PHASE5_REPOSITORY_WRITEUP.md").write_text("\n".join(writeup) + "\n", encoding="utf-8")
    repro = [
        "# Phase V reproducibility", "",
        "- Config: configs/phase5/pilot.yaml.", "- Manifest: results/phase5/manifests/pilot.jsonl.", "- Runner: kam/phase5/pilot_run.py.", "- HPG submission: scripts/submit_phase5_pilot_hpg.sh --submit.", "- Aggregation: python -m kam.phase5.pilot_aggregate --run-root results/phase5/pilot --report-root reports/phase5 --expected 144.", "- The pilot is a Stage 1 screen; Stage 2 factorial search requires review of the generated handoff.", "",
    ]
    (report_root / "PHASE5_REPRODUCIBILITY.md").write_text("\n".join(repro) + "\n", encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase V Stage 1 pilot artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/phase5"))
    parser.add_argument("--expected", type=int, default=144)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.report_root, args.expected), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
