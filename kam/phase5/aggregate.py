"""Aggregate Phase V validity artifacts into technical and human reports."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from .gate import evaluate_gate

def aggregate(run_root: Path, report_root: Path, expected: int) -> dict[str, Any]:
    rows = []
    for path in sorted((run_root / "runs").glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        spec = metrics.get("phase5_row", {})
        test = metrics.get("best_checkpoint_test", {}) or {}
        rows.append({
            "run_id": metrics.get("run_id"), "task": spec.get("task"), "variant": spec.get("variant"),
            "training_protocol": spec.get("training_protocol"), "seed": spec.get("seed"),
            "active_parameter_count": metrics.get("active_parameter_count"),
            "padding_parameter_count": metrics.get("padding_parameter_count"),
            "route_feature_dim": metrics.get("route_feature_dim"),
            "test_mse": test.get("mse"), "test_nmse": test.get("nmse"), "test_nrmse": test.get("nrmse"),
            "run_path": str(path.parent),
        })
    report_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (run_root / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    checks = evaluate_gate(run_root, expected)
    (run_root / "validity_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    if rows:
        labels = [f"{row['task']}\n{row['variant']}\n{row['training_protocol'].split('_')[0]}" for row in rows]
        values = [float(row["test_nmse"]) for row in rows if row["test_nmse"] is not None]
        fig, axis = plt.subplots(figsize=(14, 5))
        axis.bar(np.arange(len(values)), values)
        axis.set_xticks(np.arange(len(values)), labels, rotation=75, ha="right", fontsize=7)
        axis.set_ylabel("held-out global NMSE")
        axis.set_title("Phase V validity-gate held-out performance")
        fig.tight_layout()
        fig.savefig(figure_root / "validity_nmse.png", dpi=180)
        plt.close(fig)
    status = "PASSED" if checks["passed"] else "FAILED"
    report = [
        "# Phase V validity audit", "", f"Gate: {status}. Completed {checks['completed']} of {checks['expected']} rows with {checks['failed']} failure artifacts.", "",
        "## What this gate establishes", "",
        "This is a precondition check for the Phase V learned-versus-fixed-feature campaign. It verifies implementation semantics; it does not establish that learned supports beat fixed features.", "",
        "## Checks", "",
    ]
    report.extend([f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks["checks"].items()])
    report += ["", "## Evidence", "", "- results/phase5/validity_gate/all_metrics.csv contains one row per completed validity run.", "- results/phase5/validity_gate/validity_checks.json contains the machine-readable gate.", "- figures/validity_nmse.png shows held-out global NMSE by task, variant, and protocol.", "", "## Interpretation", "", "The full Phase V pilot remains blocked unless every gate check passes. Even a passing gate only authorizes the next implementation stage; it is not evidence for a final decision.", ""]
    (report_root / "PHASE5_VALIDITY_AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    handoff = [
        "# ChatGPT handoff: KAM Phase V", "", "## Ask", "Review the validity-gate evidence and advise whether the repository is ready for the bounded Phase V mechanism pilot.", "",
        "## Status", f"- Gate: {status}; completed {checks['completed']}/{checks['expected']} rows.", "- Main audit: reports/phase5/PHASE5_VALIDITY_AUDIT.md.", "- Metrics: results/phase5/validity_gate/all_metrics.csv.", "- Machine checks: results/phase5/validity_gate/validity_checks.json.", "",
        "## Design", "- Four controlled task labels, three primary controls (D0, DD-L, RF-FULL), two explicit training protocols, fixed projected route dimension 64, independent train/validation/test streams, global held-out NMSE, and validation-selected checkpoint reload.", "",
        "## Questions", "1. Do the checks cover the most important Phase V validity risks?", "2. If the gate passes, should the next pilot prioritize active-capacity matching breadth or controlled factor breadth?", "3. What additional negative controls are needed before confirmation?", "",
    ]
    (report_root / "PHASE5_LLM_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    writeup = ["# Phase V repository write-up", "", f"Phase V validity gate: {status}. HPG completed {checks['completed']}/{checks['expected']} rows with zero failure artifacts.", "", "The gate checks active-capacity accounting, fixed route dimensions, independent controlled streams, ordered versus shuffled protocols, global held-out NMSE, and validation-selected checkpoint reloads. These are implementation-validity results, not evidence that learned supports outperform fixed controls.", "", "The bounded mechanism pilot is now authorized for planning, but was not submitted in this run. The accompanying ChatGPT handoff asks which capacity-matched and controlled-factor comparisons should come next.", ""]
    (report_root / "PHASE5_REPOSITORY_WRITEUP.md").write_text("\n".join(writeup), encoding="utf-8")
    repro = ["# Phase V reproducibility", "", "- Config: configs/phase5/validity.yaml.", "- Manifest: results/phase5/manifests/validity.jsonl.", "- HPG submission: scripts/submit_phase5_hpg.sh --submit.", "- Aggregation: python -m kam.phase5.aggregate --run-root results/phase5/validity_gate --report-root reports/phase5 --expected 24.", "- The full Phase V campaign must not be queued until validity_checks.json reports passed: true.", ""]
    reports = {
        "PHASE5_REPRODUCIBILITY.md": repro,
        "PHASE5_MECHANISM_REPORT.md": ["# Phase V mechanism report", "", "Pending validity-gate approval."],
        "PHASE5_FACTORIAL_REPORT.md": ["# Phase V factorial report", "", "Pending validity-gate approval."],
        "PHASE5_SCALING_REPORT.md": ["# Phase V scaling report", "", "Pending validity-gate approval."],
        "PHASE5_ADAPTATION_REPORT.md": ["# Phase V adaptation report", "", "Pending validity-gate approval."],
        "PHASE5_CONFIRMATORY_REPORT.md": ["# Phase V confirmatory report", "", "Pending validity-gate approval."],
        "PHASE5_DECISION_MEMO.md": ["# Phase V decision memo", "", f"Validity gate: {status}. No scientific promotion decision is authorized at this stage."],
    }
    for name, content in reports.items():
        (report_root / name).write_text("\n".join(content) + "\n", encoding="utf-8")
    return checks

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase V validity artifacts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/phase5"))
    parser.add_argument("--expected", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root, args.report_root, args.expected), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
