# Phase 6 Confirmation v2 Preregistration

## Decision

This fixed-sample campaign determines whether the corrected Phase 6 signal is strong, reproducible, and scientifically valid enough to begin accelerating T-KAM-F. It also gives learned-memory lifecycle claims a separate pass/fail verdict.

No result from this campaign may be interpreted before the complete immutable graph finishes. There is no optional stopping, seed replacement, or post-hoc extension. Infrastructure failures may be rerun only with the same manifest row and seed.

## Why this campaign is needed

The corrected overnight analysis found T-KAM-F 30.4% below T-WIDE validation loss at the registered token checkpoint, but only three paired seeds were available. The old language rows also used architecture-specific data-order seeds and a very small repeated corpus. This confirmation fixes both limitations:

- every architecture in a pair shares the same training and data-order seed;
- the primary corpus is a bounded 128 MiB TinyStories V2 training slice with separate official validation-derived validation and test files;
- Tiny Shakespeare is retained only as an independent cross-corpus replication;
- the primary outcome is held-out test loss after a fixed 50M-token training budget;
- validation occurs at prespecified token checkpoints, not wall-clock checkpoints or interpolated post hoc points.

TinyStories is sourced from the public `roneneldan/TinyStories` dataset and its CDLA-Sharing-1.0 dataset card. The exact downloaded bytes and SHA-256 digests are recorded before submission.

## Fixed experiment matrix

| Cohort | Corpus | Architectures | Paired seeds | Rows | Role |
|---|---|---|---:|---:|---|
| Primary | TinyStories V2 128 MiB | T-KAM-F, T-WIDE | 30 | 60 | Confirmatory superiority |
| Secondary controls | TinyStories V2 128 MiB | T0, T-PKM | 12 | 24 | Holm-corrected context |
| Replication | Tiny Shakespeare | T-KAM-F, T-WIDE | 24 | 48 | Independent corpus replication |
| Mechanism | TinyStories V2 128 MiB | T-KAM-F, T-KAM-L, T-KAM-ALT | 8 | 24 | Learned-memory lifecycle |
| **Total** |  |  |  | **156** | Four-L4 array, throttle `%4` |

The primary sample size targets approximately 90% power at paired standardized effect `dz = 0.65`. The replication targets approximately 80% power at the same conservative planning effect. These planning effects are much smaller than the three-seed pilot effect to protect against winner's curse.

## Locked estimands and statistics

### Primary

- Estimand: paired log ratio of T-KAM-F to T-WIDE held-out test cross-entropy at 50M tokens.
- Minimum scientifically relevant improvement: 2% lower loss.
- Interval: deterministic paired bootstrap 95% confidence interval with 20,000 replicates.
- Test: two-sided paired sign-flip randomization test; 100,000 deterministic Monte Carlo permutations for 30 seeds.
- Pass: all 30 pairs present, randomization `p <= 0.05`, and the upper confidence bound is at or below `log(0.98)`.
- The primary family contains exactly one comparison, so no multiplicity adjustment is needed.

### Independent replication

- Same metric and pairing on 24 fresh Tiny Shakespeare seeds.
- Pass: all 24 pairs present, randomization `p <= 0.05`, and the upper 95% confidence bound is below zero.
- Promotion requires both the primary and replication gates.

### Secondary controls

- T-KAM-F versus T0 and T-PKM on the first 12 primary seeds.
- Holm correction is applied across this two-comparison secondary family.
- Secondary results explain the architecture landscape but cannot rescue a failed primary or replication result.

### Learned-memory lifecycle

T-KAM-L and T-KAM-ALT receive eight fresh seeds each. Every seed must show:

1. positive geometry update steps before freeze;
2. an observed nonzero key-gradient norm before freeze;
3. freeze between 79% and 81% of total tokens;
4. a frozen checkpoint in final tuning; and
5. post-freeze geometry drift no greater than `1e-10`.

This verdict is separate from fixed-key T-KAM-F promotion. A T-KAM-F result is not evidence that learned geometry helps.

## Metrics and figures

Primary and secondary metrics:

- test and validation cross-entropy;
- paired log-loss ratio, geometric relative change, median change, and win rate;
- bootstrap interval, paired randomization p-value, and standardized paired effect;
- validation-to-test generalization gap;
- total, trainable, and active parameters;
- tokens/second, wall time, estimated FLOPs, peak VRAM, and quality per GPU-hour;
- deletion/intervention metrics for registered diagnostic rows.

Lifecycle metrics:

- geometry and algebra step counts;
- memory key/value gradient norms across registered checkpoints;
- memory gate scale;
- exact freeze token and freeze fraction;
- post-freeze geometry drift;
- pre-freeze versus final-tuning validation change.

Required figures:

- paired primary held-out test loss;
- effect-size interval forest across primary, replication, and controls;
- registered-token learning curves by corpus;
- validation/test generalization diagnostics;
- learned-memory freeze timing and log-scale drift;
- matched-token resource-quality comparison.

## Validity guardrails

The final report is blocked rather than interpreted if any of these fail:

- manifest count or identity mismatch;
- missing, duplicate, failed, nonfinite, or smoke rows;
- architecture pairs with different data-order seeds;
- unequal token exposure beyond one batch increment;
- dataset checksum inconsistency within a corpus;
- split overlap or missing separate TinyStories validation/test files;
- total-parameter count outside 5% of the registered 10M target;
- unrecorded precision, environment, commit, manifest hash, or Slurm identity.

## Implementation map

- Contract: `configs/phase6/confirmation_v2.yaml`
- Manifest: `kam/phase6/confirmation_manifest.py`
- Runner: `kam/phase6/overnight_runner.py`
- Statistics/reports: `kam/phase6/confirmation_analysis.py`
- Corpus staging: `scripts/prepare_phase6_confirmation_corpora.py`
- HPG submission: `scripts/submit_phase6_confirmation_hpg.sh`
- Regression tests: `tests/test_phase6_confirmation.py`

Expected compute is approximately 105–120 L4 GPU-hours, or roughly 27–32 hours at four-way occupancy. This is intentionally larger than the screening campaign because it is the decision-grade experiment before acceleration.

## Deployed HPG campaign

- Submitted: 2026-07-28 at 13:47 UTC.
- Isolated checkout: `/blue/uf-dsi/rvalle1/KAM_confirmation_v2_20260728`
- Result root: `/blue/uf-dsi/rvalle1/KAM_confirmation_v2_results/results/phase6/confirmation_v2`
- Report root: `/blue/uf-dsi/rvalle1/KAM_confirmation_v2_results/reports/phase6/confirmation_v2`
- Log root: `/blue/uf-dsi/rvalle1/KAM_confirmation_v2_results/logs/phase6/confirmation_v2`
- Immutable manifest SHA-256: `7a47d6a54cda5e782f37c1db86081838eab8561b361bac52f57a6c3ba9f851df`
- GPU smoke array: `38203500` (5/5 rows completed, exit code 0).
- Production array: `38203848` (`0-155%4`, one L4 per row).
- Dependent final report: `38203849` (`afterany:38203848`).
- Local validation: 79/79 repository tests passed.
- HPG submission gate: 30/30 confirmation and Phase 6 tests passed.

Registered TinyStories split hashes:

- Train, 128 MiB: `7fdce72ef72919e8db78cc2580e100b4d0d17702fd70b261978480e8bad1a618`
- Validation, 11,251,300 bytes: `cd86432a3efbee3d7bc32902641ce3272830f696ddc12e15514d1f7800b23603`
- Held-out test, 11,251,301 bytes: `2ea11a0939f6e0a536e317e91aac293035886d138a0879ea4f38ee3396e32671`

A reasonable first completion check is 2026-07-29 at approximately 5:00 PM EDT. Do not inspect partial scientific effects before the final fixed sample is complete.
