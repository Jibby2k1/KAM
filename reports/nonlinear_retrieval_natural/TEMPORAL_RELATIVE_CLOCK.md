# Unit-invariant clocks for revision lookup

A signed-log clock scaled by the mean nonzero offset improved mixed-training radial lookup on the independent Dallas archive from 42.44% to 57.56%. Tripling publication offsets left all 3,504 numerical outcome records unchanged. Plain signed log moved from 42.44% to 48.46% under the same label-preserving recoding.

Both clocks are query-relative. Each token contains signed effective and publication offsets. Relative variants divide each coordinate by a scale computed independently for the current episode, then apply the same signed logarithm. The mean excludes zero offsets. The maximum variant preserves only offset ratios to the largest absolute value. Ordinal encoding preserves order and ties. One row of clock offsets can influence everybody's scale when normalization uses the entire episode.

The comparison holds calibration episodes, held-out rows, weight initialization and update budget fixed. All fits use three L4 seeds and 3,200 updates over histories of 16 or 48 records per bank. Mixed fits use half procedural correction data and half archive-shaped data. The held-out Dallas panel uses 292 groups, two cutoff dates per group, two independent symbolizations, and three seeds. Replays of the earlier linear and signed-log arms matched all 60 archived fit trajectories at the observed outcomes across the two experimental grids.

| Mixed-training radial coordinates | Procedural corrections, 48 | Corrections, 512 | Earlier Philadelphia, 48 | Dallas, 48 | Dallas publication offsets ×3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain linear | 11.72 | 1.17 | 50.57 | 8.36 | 10.22 |
| Plain signed log | 79.82 | 30.53 | 81.25 | 42.44 | 48.46 |
| Ordinal | 45.64 | 2.93 | 38.80 | 8.48 | 8.48 |
| Divide by episode maximum | 13.87 | 1.04 | 54.90 | 10.02 | 10.02 |
| Signed log after episode maximum | 79.56 | 30.60 | 80.05 | 40.55 | 40.55 |
| Signed log after episode mean | 82.81 | 39.52 | 81.98 | 57.56 | 57.56 |

Values are exact key-and-value accuracy in percent. The Dallas gain for mean relative log over plain signed log was 15.13 percentage points [9.79, 20.82], with both archive dates and symbolizations retained within each quarter and a separate fit seed resampling. The Philadelphia gain was 0.73 points [−2.40, 4.06]. For corrections at depth 512, the gains were positive in all three seed-level paired intervals. With correction-only training, relative-mean signed log reached 70.44% at depth 48 and 30.99% at depth 512, and recovered none of the Dallas answers. For archive-shaped-only training, ordinal and linear normalized radial clocks reached 53.88% and 55.96% in Dallas. The mixed fit supplies the reported balance between the procedural and archival tests.

The maximum feature difference after changing publication units was 2.38×10⁻⁷ for the episode mean and 1.19×10⁻⁷ for the episode maximum. Ordinal features were exact before inference. This test simultaneously rescales every publication offset. A future record can still distort a global scaling statistic, so isolated timestamp corruption is a separate intervention.

## One outlier per history

At a fixed candidate bank, one deterministic randomly selected fact has its publication distance from the cutoff multiplied by 64. The future condition only moves a fact that was already unavailable. The older condition only moves a past fact at a different address. Record contents and ordering stay fixed, and two independent oracles verify both target choices for every modified row. At depth 48, the future condition changes 3,444 of 3,504 rows. The wrong-address condition changes all 3,504. Median normalization uses the lower median of nonzero absolute values. The available-only mean computes scale from records that satisfy both clock cutoffs, with the clock data still visible in attention.

| Mixed radial scale, Dallas depth 48 | Clean | One unavailable outlier | One older outlier at another address |
| --- | ---: | ---: | ---: |
| Plain signed log | 42.44% | 42.47% | 42.58% |
| Relative episode mean | 57.56% | 55.88% | 56.39% |
| Relative episode median | 58.13% | 58.08% | 58.13% |
| Relative available-only mean | 56.16% | 56.08% | 49.86% |

The all-record mean lost 1.68 points [−3.29, −0.32] to an unavailable outlier. The median changed by −0.06 [−0.26, 0.17]. The available-only mean was stable in the future control and lost 6.31 [−8.52, −4.35] when the outlier was already visible under both clocks and at another address. The median kept all unit-rescaled joint counts and nearly all paired outcomes in the outlier tests, with no evidence of a clean advantage over the mean at depth 48. These are fixed-calendar, label-preserving perturbations of a real archive lookup task. [Outlier manifests and paired summaries](../../results/nonlinear_retrieval_natural/temporal_clock_outliers_v1/)

HPG L4 arrays 52994809–52994811, 52997768–52997769 and 52998883. [Ordinal and maximum normalization results](../../results/nonlinear_retrieval_natural/temporal_ordinal_clock_v1/) · [Relative signed-log results](../../results/nonlinear_retrieval_natural/temporal_relative_log_v2/) · [Dallas transfer protocol](TEMPORAL_EXTERNAL_TRANSFER.md)
