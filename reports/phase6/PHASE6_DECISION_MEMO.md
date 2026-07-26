# Phase 6 decision memo

## Decision

**No promotion decision is made yet.** Stage 0 validates the implementation and HPG execution path, but it does not establish a quality, compute, or adaptation advantage. One of the prescribed outcomes must be selected only after locked confirmation.

## Current evidence

- Local Stage 0: 128/128 pass.
- HPG Stage 0 measured pass: 128/128 pass.
- HPG Stage 1 initial profile: 64/64 structural rows pass, but the run is scientifically superseded by a geometry-factor coverage defect.
- HPG Stage 1 history-bearing profile: array `38038386` and aggregate `38038387` pass 64/64 with explicit geometry coverage, but are superseded by the task/optimizer dispatch audit.
- HPG Stage 1 task-aware replacement: array `38040026` and aggregate `38040027` pass 64/64 after the HPG package and T-WIDE readout fixes; the identity audit found zero mismatches, but short alternating schedules did not reach geometry phases.
- First full Stage 1 deployment: array `38040418` / aggregate `38040419` canceled after 1,315 partial rows for the schedule audit.
- Corrected full Stage 1 mechanism campaign: array `38042710` / aggregate `38042711` completed 3,000/3,000 rows with zero failures; independent identity/finite/dispatch and alternating-schedule audits pass. Descriptive report: `reports/phase6/PHASE6_STAGE1_FULL_REPORT.md`.
- Bounded Stage 2 transformer profile: HPG array `38049074`, aggregate `38049075` completed 48/48 rows with zero failures. The exact-manifest audit passes; total-parameter matching has 0.214% mean absolute relative error and 0.5952% maximum error. Descriptive report: `reports/phase6/PHASE6_TRANSFORMER_COMPARISON.md` and `reports/phase6/PHASE6_STAGE2_PROFILE_REPORT.md`.
- Bounded Stage 3 router profile: HPG array `38049475`, aggregate `38049476` completed 32/32 rows with zero failures. The exact-manifest audit passes; mean exact-reference recall was 0.993 exact, 0.991 chunked, 0.983 product-key, and 0.477 approximate. Descriptive report: `reports/phase6/PHASE6_ROUTER_SCALING_REPORT.md` and `reports/phase6/PHASE6_STAGE3_PROFILE_REPORT.md`.
- Bounded Stage 4 online-adaptation profile: initial run `38049583`/`38049584` exposed six nonfinite symbolic histories; after a bounded normalized-update repair, corrected run `38049769`/`38049770` completed 48/48 and passed the exact-manifest/finite-metric audit. Descriptive report: `reports/phase6/PHASE6_ADAPTATION_REPORT.md` and `reports/phase6/PHASE6_STAGE4_PROFILE_REPORT.md`.
- Bounded Stage 5 profile: initial run `38050204`/`38050205` exposed four unsupported Mackey-Glass rows; after adding the bounded dynamics fixture, corrected run `38050338`/`38050339` completed 12/12 and passed the exact-manifest audit. All rows ran only 4,096 tokens against 200M–2B declared budgets, so this is not long-training evidence. Descriptive report: `reports/phase6/PHASE6_LONG_TRAINING_REPORT.md` and `reports/phase6/PHASE6_STAGE5_PROFILE_REPORT.md`.
- Stage 6 confirmation-preparation profile: HPG array `38050441`, aggregate `38050442` completed 12/12 rows at the locked 10M budget with a passing exact-manifest audit. It uses four steps/256 tokens per row and lacks paired held-out inferential analysis, so it does not satisfy the final confirmation gate. Descriptive report: `reports/phase6/PHASE6_CONFIRMATORY_REPORT.md` and `reports/phase6/PHASE6_STAGE6_PROFILE_REPORT.md`.
- Replicated held-out quality comparisons: not run.

## Next decision gate

Review the Stage 1 frontier and bounded Stage 2–6 resource/profile evidence; do not interpret Stage 5 as convergence or Stage 6 preparation as confirmation. The decision remains **no promotion** until paired new seeds, held-out streams/corpora, inferential tests, equivalence margins, and locked confirmation satisfy the brief.
