# Phase 6 Overnight Analysis v2 Correction

## Purpose

This handoff documents the July 28, 2026 correction to the Phase 6 overnight promotion, inference, decision, adaptation, and visualization logic. It is the shortest entry point for an LLM reviewing whether the campaign supports acceleration.

## Corrected logic

- Language promotion filters to comparable `small_language` rows. Retrieval and dynamics losses cannot change the promoted language architecture.
- Production language manifests use exact matched-token budgets. Wall time, throughput, FLOPs, and VRAM are outcomes rather than stopping requirements.
- Language observations are expanded from row bundles to one record per training seed and stratified by wave, lane, task, scale, and registered token budget.
- Dynamics bundles are expanded to one record per task and training seed.
- Adaptation's tasks and schedules are averaged within each base training seed before inference.
- Pairing joins candidate and comparator by exact seed identity. Row order is irrelevant.
- Holm correction is applied within wave/lane/task/metric/comparison families.
- The final decision uses only Wave 3 matched-token quality. Screening waves cannot outvote the independent replication.
- Promotion requires at least six paired seeds, a favorable bootstrap interval, and Holm-adjusted p ≤ 0.05.

## Adapter correction

The completed legacy adaptation rows were labeled `rls` for controls and `value_only` for KAM, but `kam/phase6/overnight_runner.py` did not dispatch those adapters. It ran full-model AdamW for every architecture. Analysis v2 therefore:

1. records `adapter_declared` separately from `adapter_effective`;
2. identifies the effective method as `joint_sgd_full_model`;
3. marks it unregistered;
4. exports five base-seed observations per architecture; and
5. blocks adaptation-only promotion from those rows.

Future overnight manifests use the honest effective label until a registered adapter implementation is added.

## Primary artifacts

- `kam/phase6/overnight_manifest.py` — lane-aware promotion and matched-token manifests.
- `kam/phase6/overnight_runner.py` — exact-token stopping and adapter provenance.
- `kam/phase6/overnight_analysis.py` — normalized observations, paired statistics, decision gate, reports, and figures.
- `tests/test_phase6_components.py` — regression tests for cross-lane contamination, seed-order invariance, confirmatory power, and adaptation aggregation.

Analysis v2 adds:

- `language_seed_metrics.parquet`
- `dynamics_seed_metrics.parquet`
- `adaptation_metrics.parquet` at base-seed grain
- `adaptation_row_metrics.parquet` preserving raw row aggregates
- corrected `paired_seed_metrics.parquet`

## Interpretation boundary

The current completed Wave 3 run has only three paired language seeds. It can establish direction and magnitude, but it cannot pass the corrected confirmatory gate. A corrected result of `RETAIN_AS_DIAGNOSTIC_ONLY` means “run a clean, adequately powered confirmation before acceleration,” not “KAM failed.”

Do not use the superseded pooled-wave decision or the earlier row-position paired statistics.

## Deployed result

- HPG analysis checkout: `/blue/uf-dsi/rvalle1/KAM_analysis_v2_20260728`
- Immutable evidence view: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight_analysis_v2`
- Corrected reports: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight_analysis_v2`
- Final report job: `38201254` (`COMPLETED`, exit code 0)
- Test evidence: 72 local tests passed; 29 Phase 6 tests passed in the deployed checkout; five focused correction tests passed after the exact-token interpolation refinement.

At the exact registered Wave 3 token checkpoint, T-KAM-F had mean validation loss 2.0081 versus 2.8833 for T-WIDE (30.4% lower). There were only three exact paired seeds: the bootstrap interval favored T-KAM-F, but the exact paired permutation p-value was 0.25 and the within-family Holm-adjusted p-value was 0.75. This is promising directional evidence, not confirmatory evidence.

The corrected report package contains six descriptive figures: wave-faceted language learning curves, checkpoint-policy comparisons, common-task dynamics predictions with log error, learned-memory freeze traces, matched-budget resource/quality comparisons, and base-seed adaptation metrics. All six PNGs passed file and dimension validation; interactive visual inspection was unavailable in the local Codex sandbox because its image viewer could not initialize loopback networking.
