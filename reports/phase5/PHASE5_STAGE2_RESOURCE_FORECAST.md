# Phase V Stage 2 resource forecast

## Pre-profile evidence

The required three-row HPG pre-profiles completed successfully for each Stage 2 sub-study. The symbolic profile required one corrective retry for a 65-token shifted-input boundary and then passed.

| Sub-study | Profile job(s) | Row durations (s) | Status |
|---|---:|---:|---|
| Stage 2A component | 37986332 | 12, 12, 13 | passed |
| Stage 2B capacity | 37986333 | 11, 11, 13 | passed |
| Stage 2C factorial | 37986334 | 11, 11, 7 | passed |
| Stage 2D symbolic | 37986480 retry | 14, 12, 13 | passed after fix |

No profile row produced an out-of-memory or timeout failure. Each HPG array element requests one GPU, four CPUs, 24 GB RAM, and a four-hour limit.

## Forecast

The full campaign contains 1,590 rows. At the observed median of roughly 12 seconds per row, four-way GPU concurrency gives an idealized compute time of about 80 minutes. Allowing for queueing, larger architectures, filesystem variation, and retries, a practical wall-clock expectation is 2–4 hours after allocation. The per-row four-hour limit is conservative.

This is a planning estimate, not a result claim. The dependent aggregate jobs are the source of truth for completion and validity.
