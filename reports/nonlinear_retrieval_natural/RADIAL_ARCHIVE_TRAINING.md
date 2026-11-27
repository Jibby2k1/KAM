# From-scratch radial precision and archival revision transfer

Across three fresh seeds on B200 and L4, the combined BF16 Flash encoding transferred to archived revisions at roughly the same accuracy as eager training. At 48 records per bank, mean joint accuracy on the earlier archive panel differed from FP32 by −0.47 points on B200 and +0.26 points on L4. The later panel showed −2.08 and −0.26 points. These small sequence lengths gave little wall-time benefit from fusion.

## Matched design

The four modes trained a position-free radial decoder from identical starting weights for each seed. A procedural generator supplied revision clusters at depths 16 and 48, with a matched 3,200 updates and 13,516,800 tokens per mode. Both GPU types replayed the same training banks, update indices, initialization, and held-out recodings. The held-out cohorts kept 160 earlier and 32 later observation groups separate. Each group contributed two answerable cutoffs and two independent symbol recodings per depth.

The model was given typed addresses and time coordinates and had to select both the address and exact archived revision. The primary evaluation used the same eager FP32 forward path for all trained models. This isolates training differences. Each model was also evaluated with its native inference path. [Revision projection and prior controls](BITEMPORAL_MODEL.md#transfer-to-actual-revision-sequences) · [Philadelphia Fed archive](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-full-time-series-history)

| Device and training mode | Earlier joint, 48 | Later joint, 48 | Mean training wall time |
| --- | ---: | ---: | ---: |
| B200 eager FP32 | 88.39% | 85.16% | 16.85 s |
| B200 eager BF16 | 88.02% | 81.51% | 19.63 s |
| B200 original BF16 Flash | 88.13% | 82.55% | 18.64 s |
| B200 balanced split BF16 Flash | 87.92% | 83.07% | 18.84 s |
| L4 eager FP32 | 88.54% | 83.59% | 13.23 s |
| L4 eager BF16 | 88.07% | 83.07% | 15.34 s |
| L4 original BF16 Flash | 87.76% | 82.29% | 14.77 s |
| L4 balanced split BF16 Flash | 88.80% | 83.33% | 14.84 s |

At depth 48 the native inference path changed any cohort mean for a given mode and GPU by at most 0.52 points from common FP32. The balanced split path reduced extra peak allocated memory to 67 MB in all six runs, compared with 81–99 MB for eager FP32. Its 33- and 97-token training steps represent a different regime from the earlier 1,025-token steps that benefited strongly from Flash. The three seeds and the small later cohort leave residual uncertainty about the quality differences. The raw runs and summary give both depths, numeric exact-match rates, and seed-level counts. HPG arrays 52962403 and 52962404 ran the study. [Matched dual-device audit](../../results/nonlinear_retrieval_natural/radial_archive_training_v1/) · [Long-step timing](RADIAL_BACKWARD.md) · [Kernel encodings](RADIAL_BF16_PRECISION.md)

Recreate the two archive inputs with the vintage fixture builder using the same four source workbooks. The earlier fixture uses the builder defaults. The later fixture uses `--test-years 2023 2026`. The run files record the exact fixture hashes. [Fixture builder](../../scripts/prepare_kam_vintage_revisions.py) · [Training runner](../../scripts/run_kam_radial_archive_training.py)
