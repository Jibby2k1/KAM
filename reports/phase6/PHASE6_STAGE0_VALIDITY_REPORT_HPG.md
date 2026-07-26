# Phase 6 Stage 0 validity report

## Verdict

- Execution: **hpg**
- Rows: **128**
- Passed: **128**
- Failed: **0**
- Stage 0 gate: **PASS**
- Stage 1+ submission allowed by this report: **yes**

The gate is a correctness/system check, not evidence that sparse memory improves quality. It verifies that the implementation is safe enough to profile and compare in later stages.

## Checks

| Check | Rows | Representative result |
|---|---:|---:|
| Exact/chunked routing reference | 32 | recall@k = 1 |
| Finite backward pass | 16 | finite gradient tensors = 4 |
| Zero-gate baseline equivalence | 16 | max logit error = 0 |
| Resource accounting and timing | 16 | params = 5137; median forward = 1.17355 ms; throughput = 10413.7 tokens/s |
| Ridge/streaming solver | 16 | max direct error = 3.71231e-16 |
| Geometry rollback | 16 | trust-region rejects recorded = 1 |
| Causal masking | 16 | prefix leakage error = 0 |

## What this enables

The next safe step is the bounded Stage 1 mechanism profile using the same immutable manifest discipline and measured resource schema. Stage 2–6 arrays should remain staged behind their upstream reports. The current implementation does not yet claim support birth/death, or scientific quality/adaptation gains from any later stage.

## Reproduction

```bash
python -m kam.phase6.manifest --config configs/phase6/stage0_validity.yaml
python -m kam.phase6.run_stage0 \
  --manifest results/phase6/stage0/manifests/validity.jsonl \
  --output results/phase6/stage0/validity_results.jsonl
```

Source results: `results/phase6/stage0/hpg_runs_final/validity_results.jsonl`.
