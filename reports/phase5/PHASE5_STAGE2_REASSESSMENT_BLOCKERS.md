# Phase V Stage 2 reassessment blockers

Status checked 2026-07-25 for HPG run root `results/phase5/stage2_reassessment_5f01915`.

| Sub-study | Reassessment status | Report status |
|---|---|---|
| Stage 2A component | 450/450 valid | Passed |
| Stage 2B capacity | 476/480 metrics; 4 rows failed stream-quality validation | Blocked |
| Stage 2C factorial | 1,080/1,080 training metrics; 10 rows failed held-out stream generation | Blocked |
| Stage 2D symbolic | 60/60 trained, but all 60 failed runtime capacity-consistency checks | Blocked |

## Required fixes

1. **NARMA stability:** revise the controlled NARMA recurrence or its factor cells so streams do not saturate at the clip boundary. Stage 2B failures include a 9% clip-boundary fraction; Stage 2C held-out failures include a 59.4% clip-boundary fraction and lag-1 autocorrelation of 0.999. Re-run the affected rows and retain the stability gate.
2. **Factorial held-out completion:** the 10 Stage 2C rows have training metrics but no valid held-out metrics because held-out validation-stream generation fails. They must be rerun after the NARMA fix before aggregation.
3. **Symbolic capacity metadata:** regenerate the Stage 2D manifest after the pulled model changes. Runtime active counts no longer equal the manifest’s `resolved_active_parameters` values. For example, runtime active count 996,969 was compared with a recorded target of 1,000,212. Re-resolve architectures and rerun the affected symbolic rows; do not relax the <=1% or exact-recorded-count checks.
4. **Report dependencies:** rerun the blocked aggregate jobs only after repaired rows have complete `metrics.json`, `heldout_metrics.json`, no failure artifacts, and passing machine-readable gates.

The Stage 2 reassessment should not be used for scientific conclusions until all four sub-study reports pass.
