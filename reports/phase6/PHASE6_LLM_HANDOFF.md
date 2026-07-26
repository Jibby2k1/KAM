# Phase 6 handoff for ChatGPT feedback

## Objective

Assess whether sparse support-derived memory with fast algebra and slower geometry can improve the quality/compute or adaptation trade-off of a modern causal decoder. A negative result is useful.

## Current implementation

- `kam/transformer/`: standalone modern causal decoder with RMSNorm, SwiGLU, learned/sinusoidal positions, and optional per-block memory residuals.
- `kam/memory/`: exact and chunked top-k routing, routing diagnostics, vector/affine/low-rank experts, zero-init gates, geometry rollback, and an explicit episodic bank.
- `kam/optimization/`: ridge/Cholesky, streaming RLS, alternating schedules, trust-region state, and variable-projection fixture.
- `kam/phase6/`: immutable Stage 0 manifest, row executor, gate, resource accounting, aggregation, and report generation.
- `kam/data/phase6/`: deterministic dynamics, retrieval, and symbolic fixtures.

## Evidence so far

The local Stage 0 manifest has 128 rows: 8 checks × 2 router choices × 2 expert choices × 2 geometry choices × 2 deterministic seed tags. All 128 passed. The first corrected bounded Stage 1 HPG profile passed 64/64 structurally, but a post-run audit found task and optimizer dispatch defects; it is superseded. The task-aware replacement (`38040026` / `38040027`) passed 64/64 after a HPG package synchronization and a T-WIDE readout-shape fix; its concise report is `reports/phase6/PHASE6_STAGE1_TASKFIX_REPORT.md`. Its short fidelity did not reach alternating geometry phases, so the corrected full Stage 1 campaign (`38042710` / `38042711`) was run to completion. The retrieved 3,000-row aggregate passed; independent identity, finite-metric, dispatch, and schedule audits also passed. The detailed descriptive report is `reports/phase6/PHASE6_STAGE1_FULL_REPORT.md`.

- exact/chunked routing recall@k: 1.0;
- zero-gate baseline max-logit error: 0;
- finite backward checks: passed;
- ridge vs. direct solve max error: approximately `3.9e-16`;
- geometry trust-region/non-finite rollback: passed;
- causal-prefix leakage error: 0.

This is an implementation gate, not a scientific result. The measured HPG Stage 0 pass is complete. The corrected Stage 1 profile is a valid task/identity screening execution, and the corrected full 3,000-row mechanism campaign is complete with balanced factor coverage and schedule evidence; no quality or promotion claim is available yet.

The current tree now includes profile/full manifest builders for Stages 1–6, a resumable generic row runner, stage gates/aggregation, paired-seed statistics, matched transformer budgets, effective-capacity accounting, a real bundled small-language fixture, operational transformer ALT/VP paths, and descriptive report/plot helpers. These are executable profile infrastructure; they should not be summarized as completed quality experiments until their rows are run and aggregated.

The first architecture-aware Stage 1 HPG profile (`38036789`, aggregate `38036793`) is superseded because a manifest mapping defect left every `geometry` factor null; see `reports/phase6/PHASE6_STAGE1_GEOMETRY_GAP.md`. The history-bearing profile (`38038386` / `38038387`) then passed 64/64 but is also superseded by the task/optimizer dispatch audit; see `reports/phase6/PHASE6_STAGE1_TASK_DISPATCH_GAP.md`. The first task-aware deployment (`38039123` / `38039124`) failed before producing rows because HPG had a stale package export. The next replacement (`38039556` / `38039557`) exposed and fixed a T-WIDE readout-shape bug. The task-aware screening profile (`38040026` / `38040027`) passed its identity and finite-metric audit, but a short-budget audit found zero alternating geometry phases. The first full deployment (`38040418` / `38040419`) was canceled after 1,315 partial rows for that reason; the corrected full campaign is `38042710` / `38042711`.

The bounded Stage 2 transformer profile (`38049074` / `38049075`) then completed 48/48 rows. Its exact HPG manifest, outputs, identity audit, aggregate, and descriptive report are retained in `results/phase6/stage2_transformer_comparison/`, `reports/phase6/stage2_transformer_comparison_profile_budget1/`, and `reports/phase6/PHASE6_STAGE2_PROFILE_REPORT.md`. Total-parameter matching had 0.214% mean absolute relative error and 0.5952% maximum error. This is an execution/resource result, not a quality ranking: each row had only four steps and 256 training tokens, with no replicated held-out comparison.

The bounded Stage 3 router-scaling profile (`38049475` / `38049476`) completed 32/32 rows and passed the exact-manifest audit. Across the profile, exact/chunked/product-key/approximate routing averaged 0.993/0.991/0.983/0.477 recall against the exact reference. Product-key bank storage averaged 11.2 KB versus 2.93–3.30 MB for the other routers, but the profile mixes support sizes and precisions and does not establish a quality or scaling law. Details are in `reports/phase6/PHASE6_ROUTER_SCALING_REPORT.md` and `reports/phase6/PHASE6_STAGE3_PROFILE_REPORT.md`.

The bounded Stage 4 online-adaptation profile required one stability repair. The initial HPG run (`38049583` / `38049584`) had six nonfinite symbolic-stream histories; audit caught them even though row status was `pass`. A bounded normalized update and explicit finite-history gate were added, and the corrected full rerun (`38049769` / `38049770`) passed 48/48. It exercises A→B→A→C→A streams and emits per-row plus aggregate adaptation curves. Its pooled mean global/early/late NMSE is 8.11/8.89/7.22, but this is descriptive and not a replicated adapter comparison. Details are in `reports/phase6/PHASE6_ADAPTATION_REPORT.md` and `reports/phase6/PHASE6_STAGE4_PROFILE_REPORT.md`.

The bounded Stage 5 profile initially exposed four unsupported Mackey-Glass transformer rows (`38050204` / `38050205`). After adding the declared bounded continuous-dynamics token fixture and regression test, the corrected profile (`38050338` / `38050339`) passed 12/12. It ran only 4,096 tokens per row against 200M–2B declared budgets, so it is a dispatch/stability smoke check rather than long-training evidence. Details are in `reports/phase6/PHASE6_LONG_TRAINING_REPORT.md` and `reports/phase6/PHASE6_STAGE5_PROFILE_REPORT.md`.

The Stage 6 confirmation-preparation profile (`38050441` / `38050442`) completed 12/12 at the locked 10M total-parameter budget. It covers the three configured claim families with four rows each and passes the exact-manifest audit, but uses only four steps/256 tokens per row and has no paired held-out inferential analysis. It is preparation evidence only; no final outcome is selected. Details are in `reports/phase6/PHASE6_CONFIRMATORY_REPORT.md` and `reports/phase6/PHASE6_STAGE6_PROFILE_REPORT.md`.

## Questions for next-step advice

1. Is the Stage 1 mechanism matrix sufficiently isolated to test sparse routing, expert expressivity, geometry learning, and algebra/geometry schedules separately?
2. Which controls should be mandatory before interpreting a transformer comparison: matched active parameters, matched wall-clock, fixed random keys, dense memory tokens, MoE/PKM, or all of them?
3. Which diagnostics should be promoted to primary outcomes: quality, quality per active parameter, quality per FLOP, recall@k, routing entropy/load balance, adaptation half-life, or a composite?
4. Given the Stage 2 profile’s 0.5952% worst-case total-parameter mismatch, which controls and scales should be retained for replicated held-out comparisons?

## Recommended guardrails

Stages 2–6 bounded profiles now pass their execution/resource audits, with the Stage 4 stability repair and Stage 5 fixture repair recorded explicitly. Stage 5 remains a capped smoke check and Stage 6 remains confirmation preparation, not a final decision. Require finite metrics, measured timing/VRAM fields, exact-reference routing checks, factor identity, replicated seeds, held-out data, and reproducible manifests. Treat support visualizations as descriptive until deletion/ablation tests establish causal support use.

## Reproduction

```bash
python -m kam.phase6.manifest --config configs/phase6/stage0_validity.yaml
python -m kam.phase6.run_stage0 \
  --manifest results/phase6/stage0/manifests/validity.jsonl \
  --output results/phase6/stage0/validity_results.jsonl
```

See `reports/phase6/PHASE6_STAGE0_VALIDITY_REPORT.md` for the human-facing result.
