# Phase 6 router scaling report

**Status: bounded profile complete and audited.** HPG array `38049475` and aggregate `38049476` completed all 32 Stage 3 rows. The exact submitted manifest, outputs, identity audit, aggregate metadata, and generated descriptive report are retained under:

- `results/phase6/stage3_router_scaling/manifests/profile_hpg_38049475.jsonl`
- `results/phase6/stage3_router_scaling/hpg_runs_profile_scaling1/`
- `reports/phase6/stage3_router_scaling_profile_scaling1/`
- `reports/phase6/PHASE6_STAGE3_PROFILE_REPORT.md`

## Execution and systems checks

- 32/32 rows passed; the identity audit found no missing, extra, duplicate, mismatched, failed, nonfinite, or dispatch rows.
- All four router families were exercised across profile support caps and mixed precision settings.
- Exact-reference routing recall was measured directly; latency, throughput, peak VRAM, bank storage, effective support, dead-support, and load-balance diagnostics were emitted per row.

## Descriptive router trade-offs

Across the eight rows per router family, the arithmetic means were:

| router | recall@k vs exact | routing ms | tokens/s | bank storage | peak VRAM |
|---|---:|---:|---:|---:|---:|
| exact | 0.993 | 96.1 | 170.7 | 3.30 MB | 19.1 MB |
| chunked | 0.991 | 113.3 | 146.9 | 3.23 MB | 18.8 MB |
| product-key | 0.983 | 119.8 | 148.0 | 11.2 KB | 21.9 MB |
| approximate | 0.477 | 116.0 | 144.4 | 2.93 MB | 15.2 MB |

The profile therefore verifies two useful implementation-level facts: chunking preserved near-exact routing in these bounded rows, and product-key routing substantially reduced stored-bank bytes while retaining high average reference recall. Approximate routing produced a materially lower average reference recall and should not be treated as an interchangeable systems control without a quality-recovery or accuracy-budget analysis. The toy profile also shows exact routing faster than approximate routing on these particular GPU shapes; this is a measured local result, not a general asymptotic claim.

These means mix support sizes, precisions, and top-k values by design. They are screening summaries, not a scaling law or a paired treatment effect. The next step is to use the routing evidence to define a small replicated set for online adaptation and held-out quality evaluation; no router is promoted from this profile alone.
