# Phase II Final Experiment Report

## Technical summary

The five-seed paired grid completed **50 runs** across `mackey_glass, narma` and `D0, DD, DR, R0, RR` using seeds `7, 11, 19, 23, 31`. No Holm-adjusted comparison has a confidence interval excluding zero; the current evidence does not justify a mechanism claim or promotion to ten-seed confirmation.

The largest positive paired MSE improvement is the mean baseline-minus-variant difference below. Positive means the candidate had lower loss.

## Paired inference

| Task | Baseline | Candidate | Pairs | Mean improvement | 95% bootstrap CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| mackey_glass | D0 | R0 | 5 | 0.000198515 | [-4.12655e-05, 0.000381465] | 1 |
| mackey_glass | DD | RR | 5 | 4.31711e-05 | [-0.000102875, 0.000168107] | 1 |
| mackey_glass | DD | DR | 5 | 2.35767e-05 | [-3.75007e-06, 4.94395e-05] | 1 |
| narma | D0 | R0 | 5 | -0.0166418 | [-0.0503502, 0.0337516] | 1 |
| narma | DD | RR | 5 | -0.0264436 | [-0.0650398, 0.0129321] | 1 |
| narma | DD | DR | 5 | 0.00286424 | [-0.00353981, 0.0083133] | 1 |

## What the evidence says

- The one-seed screen suggested R0 on Mackey–Glass and RR on NARMA, but the five-seed paired analysis is inconclusive for all registered comparisons.
- The Phase A intervention path is implemented and has been run on the ten one-seed screen checkpoints; it reports branch deltas, key/value perturbations, support utilization, frozen probes, and top/random/bottom deletions.
- The paired grid uses shared task/seed generation across variants, but it is not parameter-matched and does not yet include switching post-shift metrics, support-regime alignment, or confirmatory ten-seed inference.

## Decision gates

Do not promote a radial-memory claim until the candidate passes all pre-registered gates: stationary degradation ≤5%, post-shift improvement ≥15%, corrected CI excluding zero, top-support deletion stronger than random, noncollapsed support use, and justified time/parameter overhead.

## Recommended next step

Run the smallest five-seed screening set that adds switching Mackey–Glass/NARMA, exact parameter matching, and prequential predict→score→reveal→update with frozen/NLMS/SGD/RLS adapters. Only promote a comparison to ten confirmatory seeds if it passes the screen gates.

## Artifacts

- Paired metrics: `results/phase2/paired_screen/all_metrics.csv`
- Paired inference: `results/phase2/paired_screen/paired_stats.csv`
- One-seed intervention table: `results/phase2/reanalysis_metrics.csv`
- Raw intervention JSON: `results/phase2/reanalysis_raw/`
- Prequential smoke: `results/phase2/prequential_metrics.csv`
