# Phase 6 overnight report index

Status: timeout repair queued; no scientific conclusion is available yet.

The initial four-L4 campaign passed preflight and Stage 1 reanalysis, then produced 20/32 valid Wave 1 rows before 12 infrastructure timeouts. The selective repair preserves those 20 rows and runs from array `38087856` to final report job `38087863`. See `docs/codex/KAM_PHASE6_OVERNIGHT_TIMEOUT_REPAIR.md`.

Read after completion:

1. `OVERNIGHT_EXECUTION_REPORT.md`
2. `OVERNIGHT_DECISION_MEMO.md`
3. `OVERNIGHT_LANGUAGE_REPORT.md`
4. `OVERNIGHT_DYNAMICS_REPORT.md`
5. `OVERNIGHT_OPTIMIZATION_REPORT.md`
6. `OVERNIGHT_ADAPTATION_REPORT.md`
7. `OVERNIGHT_REPRODUCIBILITY.md`
8. `STAGE1_FRONTIER_REANALYSIS.md`

Machine-readable evidence is under `results/phase6/overnight/`. The final report job overwrites the queued placeholders with evidence-backed reports. Do not infer an outcome from the placeholders, Slurm completion, or the existence of checkpoints.

For implementation details and exact commands, read `docs/codex/KAM_PHASE6_OVERNIGHT_IMPLEMENTATION_GUIDE.md`.
