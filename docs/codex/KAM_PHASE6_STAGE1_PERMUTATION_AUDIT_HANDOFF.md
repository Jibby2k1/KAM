# Phase 6.2 Stage 1 checkpoint permutation audit

## Purpose

Stage 1 completed all 168 training rows, but the preregistered validity gate failed: two rows failed the strict FP32 matched key/expert permutation identity check and 118 rows failed the BF16 operational tolerance. This audit diagnoses those measurements from saved final checkpoints. It does not retrain models, revise thresholds, or create new inferential evidence.

## Locked sample

The manifest includes every strict FP32 failure plus, within each Stage 1 arm, the BF16 closest failure, worst failure, and closest pass. Selection distance is the maximum of the top-1 flip rate and predictive KL after dividing each by its registered tolerance. Duplicate checkpoints are audited once and retain all selection roles.

## Measurements

Each selected checkpoint receives three baseline repeats, the original full matched permutation plus two additional permutation seeds, a matched permutation of each memory layer in isolation, and per-layer routing-margin quantiles. The exact Stage 1 anchor seed and first four anchor sequences are reused.

Registered tolerances remain fixed at `2e-5` maximum absolute FP32 logit difference, `2e-2` BF16 top-1 flip rate, and `1e-3` BF16 predictive KL.

## Conditional interpretation

- `STRICT_SEMANTIC_FAILURE_REPRODUCED`: localize the failing layer(s) and reproduce those checkpoints/seeds in a narrow clean-environment replication before any efficacy claim.
- `BF16_PERMUTATION_ORDER_SENSITIVITY_REPRODUCED`: treat the BF16 gate as an implementation-order sensitivity; retain the failed registered gate and design a prospective precision-robust validity criterion for a new campaign.
- `ORIGINAL_FAILURES_NOT_REPRODUCED`: investigate execution provenance and determinism before deciding whether a narrow replication is justified.
- `AUDIT_EXECUTION_OR_REPEATABILITY_BLOCKED`: repair the audit itself; do not interpret Stage 1 efficacy.

No Phase 6.2 Stage 2 launch is authorized by this audit alone.
