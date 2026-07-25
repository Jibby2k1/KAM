"""Aggregate Stage 2A–2D runs into reports and paired effects."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from .stage2_gate import evaluate_gate
from .stage2_stats import paired_effects, write_effects


def aggregate(run_root: Path, report_root: Path, expected: int, stage_name: str) -> dict[str, Any]:
    paths = sorted((run_root / "runs").glob("*/metrics.json"))
    rows = []
    for path in paths:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = metrics.get("phase5_row", {})
        test = metrics.get("best_checkpoint_test", {}) or {}
        heldout = metrics.get("heldout_stream_metrics", [])
        heldout_nmse = [float(item["nmse"]) for item in heldout if item.get("nmse") is not None]
        heldout_cross_entropy = [float(item["cross_entropy"]) for item in heldout if item.get("cross_entropy") is not None]
        primary_values = heldout_nmse or heldout_cross_entropy
        primary_name = "nmse" if heldout_nmse else "cross_entropy" if heldout_cross_entropy else "unknown"
        rows.append({
            "run_id": metrics.get("run_id"), "task": row.get("task"), "variant": row.get("variant"),
            "cell": row.get("cell"), "seed_index": row.get("seed_index"), "stage": stage_name,
            "target_active_parameters": row.get("target_active_parameters"),
            "active_parameter_count": metrics.get("active_parameter_count"),
            "active_capacity_match_error": metrics.get("active_capacity_match_error"),
            "padding_parameter_count": metrics.get("padding_parameter_count"),
            "task_type": row.get("task_type"), "test_nmse": test.get("nmse"), "test_nrmse": test.get("nrmse"),
            "test_cross_entropy": test.get("cross_entropy"),
            "heldout_nmse": statistics.fmean(heldout_nmse) if heldout_nmse else None,
            "heldout_primary_metric": statistics.fmean(primary_values) if primary_values else None,
            "heldout_metric_name": primary_name,
            "heldout_streams": len(heldout), "total_seconds": metrics.get("total_seconds"),
            "return_probability": row.get("return_probability"), "regime_separation": row.get("regime_separation"),
            "observability": row.get("observability"), "center_initialization": metrics.get("center_initialization"),
            "key_trainable": metrics.get("key_trainable"), "value_trainable": metrics.get("value_trainable"),
            "query_path_trainable": metrics.get("query_path_trainable"), "score_path_trainable": metrics.get("score_path_trainable"),
            "memory_output_trainable": metrics.get("memory_output_trainable"), "backbone_trainable": metrics.get("backbone_trainable"),
            "route_mode": metrics.get("route_mode"), "run_path": str(path.parent),
        })
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (run_root / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    effects = paired_effects(rows)
    write_effects(run_root / "paired_effects.csv", effects)
    checks = evaluate_gate(run_root, expected, require_distinct_task_generators=stage_name != "stage2D_symbolic")
    (run_root / "stage2_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("heldout_primary_metric")
        if value is not None:
            grouped[(str(row.get("task")), str(row.get("variant")))].append(float(value))
    summary = [{"task": task, "variant": variant, "n": len(values), "mean_heldout_metric": statistics.fmean(values), "sd_heldout_metric": statistics.stdev(values) if len(values) > 1 else 0.0} for (task, variant), values in sorted(grouped.items())]
    with (run_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["task", "variant"])
        writer.writeheader()
        writer.writerows(summary)
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    if summary:
        labels = [f"{item['task']}\n{item['variant']}" for item in summary]
        values = [item["mean_heldout_metric"] for item in summary]
        fig, axis = plt.subplots(figsize=(14, 5))
        axis.bar(range(len(values)), values)
        axis.set_xticks(range(len(values)), labels, rotation=75, ha="right", fontsize=7)
        axis.set_ylabel("mean held-out primary metric")
        axis.set_title(f"Phase V {stage_name}: held-out performance")
        fig.tight_layout()
        fig.savefig(figure_root / f"{stage_name}_heldout_nmse.png", dpi=180)
        plt.close(fig)
    status = "PASSED" if checks["passed"] else "FAILED"
    names = {"stage2A_component": "COMPONENT", "stage2B_capacity": "CAPACITY", "stage2C_factorial": "FACTORIAL", "stage2D_symbolic": "SYMBOLIC"}
    label = names.get(stage_name, stage_name.upper())
    report = [
        f"# Phase V Stage 2 {label} report", "",
        f"Status: {status}. Completed {checks['completed']}/{checks['expected']} rows with {checks['failed']} failure artifacts.", "",
        "Held-out streams are aggregated within training seed. The report is descriptive until paired effects, bootstrap intervals, permutation tests, and Holm adjustment are reviewed.", "",
        "## Summary", "",
    ]
    report.extend([f"- {item['task']} / {item['variant']}: mean held-out primary metric={item['mean_heldout_metric']:.5g}, SD={item['sd_heldout_metric']:.5g}, n={item['n']}" for item in summary])
    report += ["", "## Evidence", "", f"- results/phase5/stage2/{stage_name}/all_metrics.csv", f"- results/phase5/stage2/{stage_name}/paired_effects.csv", "- stage2_checks.json", "", "## Guardrail", "", "No Stage 3 scaling, online-adaptation confirmation, or natural-language conclusion is authorized from this report alone."]
    report_root.joinpath(f"PHASE5_STAGE2_{label}_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/phase5"))
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--stage-name", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.report_root, args.expected, args.stage_name), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
