# Phase 6 Stage 1 mechanism profile — superseded task-dispatch run

> **Status: superseded for scientific interpretation.** A post-run audit found that MQAR rows fell through to a dynamics fixture and optimizer labels were not operational. The outputs remain useful for execution/resource debugging only. The first task-aware deployment (`38039123` / `38039124`) failed uniformly because HPG had a stale data-package export; the next replacement (`38039556` / `38039557`) exposed a T-WIDE readout-shape bug. The passing task-aware profile is `38040026` / `38040027`.

The passing task-aware replacement is reported in [`PHASE6_STAGE1_TASKFIX_REPORT.md`](PHASE6_STAGE1_TASKFIX_REPORT.md) for HPG array `38040026` / aggregate `38040027`.

## Technical summary

The corrected HPG profile completed **64/64 rows successfully** with finite metrics, unique row identities, exact manifest identity, and explicit coverage of all six declared geometry modes. This is a valid implementation/profile result, not a promotion result: the profile is far smaller than the 3,000-job Stage 1 target, uses a Latin-hypercube screen rather than replicated inferential cells, and runs a short synthetic regression fixture.

The descriptive signal is useful for pruning. Mean final loss was `0.02229` versus `0.03025` at initialization, a mean per-row relative reduction of `31.0%`. `T-WIDE` and vector-value rows had the lowest descriptive losses in this screen; `T-KAM-L` was the strongest KAM architecture by mean loss. These comparisons are confounded by the unbalanced profile and unmatched cost, so they do not establish a KAM advantage.

## Execution and coverage

| Item | Result |
|---|---:|
| HPG array / aggregate | `38038386` / `38038387` |
| Rows / pass / fail | `64 / 64 / 0` |
| Device | CUDA for all rows |
| Geometry modes | 6; 10–11 rows each |
| Architectures | T0, T-WIDE, T-MEMTOK, T-KAM-F, T-KAM-L |
| Tasks | prototype, switching Mackey–Glass, switching NARMA, MQAR |
| Fidelities | 5%, 20%, 50%, 100% |
| Manifest identity mismatches | 0 |
| Non-finite metric values | 0 |
| True Parquet artifacts | 11; produced with `pyarrow` on HPG |

The prior HPG profile (`38036789` / `38036793`) is retained as a [superseded geometry-coverage run](PHASE6_STAGE1_GEOMETRY_GAP.md); it must not be combined with this result.

## Descriptive architecture comparison

Means below are descriptive across the rows assigned to each factor; they are not paired estimates.

| Architecture | n | Mean final loss | Mean relative reduction | Mean forward ms | Mean active parameters |
|---|---:|---:|---:|---:|---:|
| T0 | 13 | 0.02978 | 13.5% | 0.033 | 1,056 |
| T-WIDE | 13 | 0.01143 | 45.5% | 0.051 | 4,192 |
| T-MEMTOK | 13 | 0.02744 | 20.4% | 4.120 | 3,814 |
| T-KAM-F | 12 | 0.02768 | 27.5% | 4.129 | 13,292 |
| T-KAM-L | 13 | 0.01552 | 48.1% | 4.101 | 13,456 |

The profile does not match active parameters, FLOPs, or wall-clock across these rows. In particular, KAM rows are much slower than the tiny dense controls in this implementation profile. The result therefore supports further profiling and pruning, not a claim of conditional-compute efficiency.

## Geometry, optimization, and expert signals

| Factor | Lowest descriptive mean loss | Mean final loss | n |
|---|---|---:|---:|
| Geometry | fixed k-means | 0.01754 | 11 |
| Geometry | learned low-rank delta | 0.02101 | 11 |
| Geometry | fixed data sample | 0.02149 | 11 |
| Optimizer | alternating 8:1 | 0.01708 | 8 |
| Optimizer | variable projection implicit | 0.02030 | 8 |
| Expert | vector | 0.01110 | 22 |

These are screening associations, not causal effects: geometry, optimizer, expert, task, architecture, support count, and fidelity are not fully crossed in the 64-row profile. The full Stage 1 design must use replicated seeds and matched comparisons before promotion or pruning.

## Support diagnostics and resource evidence

Among 38 rows that emitted sparse-routing diagnostics, mean routing entropy was `2.079`, mean effective support count was `9.36`, mean dead-support fraction was `0.625`, duplicate fraction was `0`, and mean tokens per support was `20.61`. The wide dead-support range (`0`–`0.953`) is a reason to retain support-utilization and load-balance metrics as primary pruning signals.

Across all rows, mean measured forward time was `2.461 ms` (range `0.0295`–`4.401 ms`), mean throughput was about `987k` tokens/s, mean peak allocated VRAM was `17.7 MiB`, and mean active/total parameter counts were `7,066`/`8,378`. These values are profile measurements on the HPG CUDA environment, not scale laws.

Generated visuals include:

- [`learning_curves.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_final/learning_curves.png), now computed from per-step loss histories;
- [`memory_diagnostics.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_final/memory_diagnostics.png);
- [`router_load.png`](../../results/phase6/stage1_mechanism/hpg_runs_profile_final/router_load.png);
- per-row prediction-versus-true, signed-error, and log absolute-error plots beneath each row directory.

## Limitations and next action

This profile does not provide independent replicated seeds per cell, held-out schedule evaluation, matched-compute comparisons, deletion/ablation evidence for causal support use, or the 3,000–6,000 completed/pruned jobs specified for the full mechanism grid. It also uses the small mechanism fixture rather than the Stage 2 transformer-scale comparison.

The next safe action is to review the profile frontier and launch the corrected full Stage 1 manifest (`3,000` rows) with the same immutable-factor audit. Stage 2 transformer comparisons should remain gated behind that review; no configuration should be promoted from this profile alone.

## Reproducibility

Canonical local outputs are under [`results/phase6/stage1_mechanism/hpg_runs_profile_final/`](../../results/phase6/stage1_mechanism/hpg_runs_profile_final/). The HPG aggregate receipt and report are under [`reports/phase6/PHASE6_STAGE1_HPG/`](PHASE6_STAGE1_HPG/). The manifest provenance is recorded in `results/phase6/stage1_mechanism/manifests/profile_summary.json` and `full_summary.json`.
