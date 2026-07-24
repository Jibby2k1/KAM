"""Evaluate the Phase V validity gate from run artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def evaluate_gate(run_root: str | Path, expected: int) -> dict[str, Any]:
    run_root = Path(run_root)
    metrics_paths = sorted((run_root / "runs").glob("*/metrics.json"))
    failure_paths = sorted((run_root / "runs").glob("*/failure.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    checks = {
        "expected_rows_complete": len(metrics) == expected,
        "no_failure_artifacts": not failure_paths,
        "zero_padding_primary_rows": all(int(row.get("padding_parameter_count", 0)) == 0 for row in metrics),
        "fixed_route_dimension": bool(metrics) and all(int(row.get("route_feature_dim", -1)) == 64 for row in metrics),
        "best_checkpoint_test_reload": bool(metrics) and all(row.get("best_checkpoint_test") is not None for row in metrics),
        "global_nmse_present": bool(metrics) and all(row.get("best_checkpoint_test", {}).get("nmse") is not None for row in metrics),
        "independent_split_streams": bool(metrics) and all(row.get("data_metadata", {}).get("independent_split_streams") for row in metrics),
        "row_validity_checks": bool(metrics) and all(all(row.get("phase5_validity_checks", {}).values()) for row in metrics),
    }
    return {"expected": expected, "completed": len(metrics), "failed": len(failure_paths), "checks": checks, "passed": all(checks.values())}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_gate(args.run_root, args.expected), indent=2, sort_keys=True))
