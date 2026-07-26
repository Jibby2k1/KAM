from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .gates import evaluate_stage0_results


def _load(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_stage0_report(results_path: str | Path, output_path: str | Path, *, execution: str = "local") -> dict[str, Any]:
    rows = _load(results_path)
    gate = evaluate_stage0_results(rows)
    metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "pass":
            metrics[str(row["task"])].append(row.get("metrics", {}))

    def mean(task: str, key: str) -> str:
        values = [float(item[key]) for item in metrics.get(task, []) if key in item]
        return f"{sum(values) / len(values):.6g}" if values else "n/a"

    report = f"""# Phase 6 Stage 0 validity report

## Verdict

- Execution: **{execution}**
- Rows: **{gate['row_count']}**
- Passed: **{gate['status_counts'].get('pass', 0)}**
- Failed: **{sum(value for key, value in gate['status_counts'].items() if key != 'pass')}**
- Stage 0 gate: **{'PASS' if gate['stage0_pass'] else 'BLOCKED'}**
- Stage 1+ submission allowed by this report: **{'yes' if gate['large_stage_submission_allowed'] else 'no'}**

The gate is a correctness/system check, not evidence that sparse memory improves quality. It verifies that the implementation is safe enough to profile and compare in later stages.

## Checks

| Check | Rows | Representative result |
|---|---:|---:|
| Exact/chunked routing reference | {len(metrics.get('route_exact_reference', [])) + len(metrics.get('route_chunked_reference', []))} | recall@k = {mean('route_exact_reference', 'recall_at_k')} |
| Finite backward pass | {len(metrics.get('gradient_finite', []))} | finite gradient tensors = {mean('gradient_finite', 'gradient_tensors')} |
| Zero-gate baseline equivalence | {len(metrics.get('zero_gate_equivalence', []))} | max logit error = {mean('zero_gate_equivalence', 'max_logit_error')} |
| Resource accounting and timing | {len(metrics.get('resource_accounting', []))} | params = {mean('resource_accounting', 'total_parameters')}; median forward = {mean('resource_accounting', 'forward_median_ms')} ms; throughput = {mean('resource_accounting', 'throughput_tokens_per_sec')} tokens/s |
| Ridge/streaming solver | {len(metrics.get('ridge_solver', []))} | max direct error = {mean('ridge_solver', 'direct_solve_error')} |
| Geometry rollback | {len(metrics.get('geometry_rollback', []))} | trust-region rejects recorded = {mean('geometry_rollback', 'trust_region_rejected')} |
| Causal masking | {len(metrics.get('causal_mask', []))} | prefix leakage error = {mean('causal_mask', 'prefix_future_change_error')} |

## What this enables

The next safe step is the bounded Stage 1 mechanism profile using the same immutable manifest discipline and measured resource schema. Stage 2–6 arrays should remain staged behind their upstream reports. The current implementation does not yet claim support birth/death, or scientific quality/adaptation gains from any later stage.

## Reproduction

```bash
python -m kam.phase6.manifest --config configs/phase6/stage0_validity.yaml
python -m kam.phase6.run_stage0 \\
  --manifest results/phase6/stage0/manifests/validity.jsonl \\
  --output results/phase6/stage0/validity_results.jsonl
```

Source results: `{results_path}`.
"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a concise Phase 6 Stage 0 report")
    parser.add_argument("--results", type=Path, default=Path("results/phase6/stage0/validity_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase6/PHASE6_STAGE0_VALIDITY_REPORT.md"))
    parser.add_argument("--execution", default="local")
    args = parser.parse_args()
    print(json.dumps(build_stage0_report(args.results, args.output, execution=args.execution), indent=2))


if __name__ == "__main__":
    main()
