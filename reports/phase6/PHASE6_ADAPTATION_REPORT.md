# Phase 6 online adaptation report

**Status: bounded profile complete and audited after a stability repair.** The first HPG Stage 4 profile (`38049583` / `38049584`) executed 48 rows but exposed six nonfinite symbolic-stream histories under the audit. The raw online update was replaced with a bounded normalized update, the six affected rows passed locally, and the full corrected profile (`38049769` / `38049770`) completed 48/48 with a clean identity and finite-metric audit.

The corrected evidence is retained under:

- `results/phase6/stage4_online_adaptation/manifests/profile_hpg_38049769.jsonl`
- `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt2/`
- `reports/phase6/stage4_online_adaptation_profile_adapt2/`
- `reports/phase6/PHASE6_STAGE4_PROFILE_REPORT.md`

## Execution and adaptation checks

- The corrected rows use the prescribed A→B→A→C→A stream schedule, 240 online steps, four task fixtures, six architecture paths, and eight adapter choices.
- 48/48 rows passed; no row had missing, extra, duplicate, mismatched, failed, nonfinite, or dispatch errors.
- The aggregate profile mean global NMSE was 8.11, early NMSE 8.89, late NMSE 7.22, and reacquisition index 13.5 steps. These pooled means mix tasks and adapters and are not a treatment estimate.
- The profile emitted per-row squared-error histories, row-level `online_adaptation_curve.png` files, and an aggregate `adaptation_curves.png`; geometry-update counts and episodic-use indicators are reported alongside quality metrics.

## Descriptive observations and limitation

RLS/NLMS rows often show much lower late error on the bounded dynamics fixtures, while persistent-memory, episodic, and geometry-update rows expose distinct reacquisition and stability behavior. The symbolic stability repair is itself a useful implementation finding: unnormalized online updates silently produced `inf` histories even though row status was previously marked `pass`. The explicit finite-history gate now prevents that failure from being treated as evidence.

This remains a descriptive profile, not evidence for a general adaptation advantage. The rows do not provide replicated paired seeds, held-out stream families, or a locked adapter equivalence analysis. Use the profile to select a small, stable set of adapters and memory controls for confirmatory online evaluation; make no promotion decision here.
