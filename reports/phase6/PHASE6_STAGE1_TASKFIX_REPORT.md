# Phase 6 Stage 1 task-aware screening report

## Status

This is a passing **64-row HPG task-aware screening profile**, not the prescribed 3,000–6,000-row mechanism campaign and not confirmatory evidence. Its manifest/task/optimizer identity is valid, but its short fidelity did not reach geometry phases for alternating schedules; do not use its optimizer or geometry means as final evidence. The corrected full campaign completed as array `38042710` / aggregate `38042711`; see `PHASE6_STAGE1_FULL_REPORT.md` for the final mechanism screen.

| Item | Result |
|---|---:|
| HPG array / aggregate | `38040026` / `38040027` |
| Rows / pass / fail | `64 / 64 / 0` |
| Tasks | 16 each: MQAR, prototype, switching Mackey–Glass, switching NARMA |
| Architectures | T0, T-WIDE, T-MEMTOK, T-KAM-F, T-KAM-L |
| Optimizer labels | 8 rows each across all 8 declared modes |
| Geometry modes | 6; 10–11 rows each |
| Manifest identity mismatches | 0 |
| Nonfinite metrics | 0 |
| Parquet artifacts | 11; verified as Apache Parquet |

The retrieved artifacts are under [`results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/`](../../results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/). The machine-readable aggregate is [`stage1_mechanism_aggregate.json`](stage1_mechanism_taskfix3/stage1_mechanism_aggregate.json), and the independent identity audit is [`identity_audit.json`](stage1_mechanism_taskfix3/identity_audit.json). The reusable audit command is `python3 scripts/audit_phase6_run.py`.

## Dispatch and validity audit

Every row matched its manifest on task, optimizer, architecture, expert, geometry, seed, fidelity, model width, support count, and top-k. The recorded `metrics.optimizer_mode` also matched the declared optimizer for all 64 rows. MQAR rows used the retrieval fixture and had a valid-target mask fraction of `1/24 = 0.0417`; their losses are therefore evaluated only on the retrieval targets, not on padded positions.

The profile passed the execution gate after fixing two issues found in earlier runs: explicit task dispatch for MQAR and dynamics, and operational optimizer-specific paths. A subsequent audit found that the short profile did not enter alternating geometry phases; the runner now guarantees and records at least one geometry update when geometry parameters exist. See [`PHASE6_STAGE1_TASK_DISPATCH_GAP.md`](PHASE6_STAGE1_TASK_DISPATCH_GAP.md).

## Descriptive quality results

The mean initial loss was `0.02739`; mean final loss was `0.01343`. The mean per-row relative reduction was `55.9%` (median `54.4%`). These are descriptive summaries over an unbalanced screening design.

| Factor | Mean final loss | Mean relative reduction |
|---|---:|---:|
| prototype | 0.00349 | 66.1% |
| MQAR | 0.00966 | 61.0% |
| switching NARMA | 0.01784 | 51.0% |
| switching Mackey–Glass | 0.02273 | 45.7% |

| Optimizer | Mean final loss | Mean relative reduction |
|---|---:|---:|
| variable projection, stop-grad | 0.00073 | 96.6% |
| variable projection, implicit | 0.00097 | 96.4% |
| ridge resolve | 0.00090 | 95.9% |
| alternating 8:1 | 0.01731 | 36.5% |
| dictionary update | 0.02027 | 28.5% |
| joint SGD | 0.02184 | 31.7% |
| alternating 32:1 | 0.01989 | 42.8% |
| alternating 128:1 | 0.02553 | 19.1% |

The solve and variable-projection rows are expected to be much stronger on this short regression fixture because they directly fit a readout. They should be treated as algebra-path controls, not evidence that a KAM geometry is superior. Architecture and geometry means are likewise not matched on active parameters, FLOPs, or wall-clock; the low T-WIDE loss and very low measured time are not a fair memory comparison.

## Memory and systems diagnostics

Across the 38 rows that instantiated sparse memory, mean routing entropy was `2.079`, mean effective support count was `9.35`, mean dead-support fraction was `0.704`, and mean load-balance error was `13.68`. This suggests that the short profile did not use all declared supports uniformly. It is a diagnostic signal requiring support-ablation and matched-capacity follow-up, not a failure of the implementation gate.

Across all rows, measured forward time averaged `2.42 ms` and peak allocated memory averaged `15.6 MB`, but the range was broad and timing was not resource-matched across architectures. Training histories contain a mean of three points because profile fidelity is intentionally short.

The HPG run includes [`learning_curves.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/learning_curves.png), [`memory_diagnostics.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/memory_diagnostics.png), [`router_load.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/router_load.png), and per-row prediction/true/error figures.

## Recommended next steps

1. Review the full Stage 1 design before submission; require balanced cells and replicated seeds for task × architecture × optimizer × geometry claims.
2. Add matched active-parameter, FLOP, wall-clock, and peak-memory controls before interpreting architecture quality.
3. Promote support deletion, support permutation, routing ablation, and geometry-free controls to primary diagnostics.
4. Keep solve-based optimizers separate from joint/alternating geometry comparisons, and report quality per unit compute.
5. After the full Stage 1 aggregate and schedule audits pass, use its predeclared frontier to select a small Stage 2 transformer comparison.
