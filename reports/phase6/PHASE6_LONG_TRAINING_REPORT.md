# Phase 6 long-training report

**Status: bounded profile complete and audited; long-budget evidence not established.** The first Stage 5 HPG deployment (`38050204` / `38050205`) exposed four unsupported `switching_mackey_glass` transformer rows. The dispatcher now includes a bounded 64-token quantized dynamics fixture, the repair is covered by a regression test, and the corrected profile (`38050338` / `38050339`) completed 12/12 with a clean audit.

Corrected artifacts are retained under:

- `results/phase6/stage5_long_training/manifests/profile_hpg_38050338.jsonl`
- `results/phase6/stage5_long_training/hpg_runs_profile_long2/`
- `reports/phase6/stage5_long_training_profile_long2/`
- `reports/phase6/PHASE6_STAGE5_PROFILE_REPORT.md`

## What was actually measured

- Tasks: `small_language`, `prototype`, and `switching_mackey_glass`; architectures: `T0`, `T-MOE`, `T-KAM-F`, and `T-KAM-L`; scales: 10M and 30M.
- All 12 rows ran 64 steps and exactly 4,096 training tokens.
- Declared token budgets ranged from 200M to 2B, so budget completion was only `2.05e-6` to `2.05e-5` (mean `9.78e-6`).
- Total-parameter matching had 0.144% mean absolute relative error and 0.522% maximum error.
- The resulting losses/perplexities and learning curves are useful for smoke-level stability and dispatch checks, not for convergence, scaling, or long-training comparisons.

Do not promote a model from this profile. A genuine Stage 5 run requires a justified larger token budget after Stage 2–4 review, with resource monitoring and held-out evaluation.
