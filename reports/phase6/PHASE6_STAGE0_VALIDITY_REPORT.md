# Phase 6 Stage 0 validity report

## Verdict

- Execution: **local**
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
| Resource accounting and timing | 16 | params = 5137; median forward = 0.998679 ms; throughput = 12493.9 tokens/s |
| Ridge/streaming solver | 16 | max direct error = 3.92048e-16 |
| Geometry rollback | 16 | trust-region rejects recorded = 1 |
| Causal masking | 16 | prefix leakage error = 0 |

## What this enables

The next safe step is a small HPG profile using the same immutable manifest and environment, followed by measured timing/VRAM checks. Only after those results are recorded should the Stage 1–6 arrays be expanded. The current implementation does not yet claim support birth/death, approximate routing, a full product-key memory path, or the complete online-adaptation campaign.

## Reproduction

```bash
python -m kam.phase6.manifest --config configs/phase6/stage0_validity.yaml
python -m kam.phase6.run_stage0 \
  --manifest results/phase6/stage0/manifests/validity.jsonl \
  --output results/phase6/stage0/validity_results.jsonl
```

Source results: `results/phase6/stage0/validity_results.jsonl`.
