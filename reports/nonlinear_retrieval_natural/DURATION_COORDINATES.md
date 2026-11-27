# Position and clock coordinates for temporal pointers

Normalizing the numeric inputs and the position coordinates together raised exact original-and-intervention recovery on authored schedule edits from 51.27% to 62.92%. Each change on its own lowered that measure. A wider token-matched training schedule improved generalization to 256 events and recovered fewer authored edits than the normalized short schedule.

## Matched controls

Six modes used the same dot-context architecture and 2,400 updates totaling 2,572,800 tokens. Each seed generated records for a short 16/32-event curriculum and a wide 16/32/64/128-event curriculum. Within each seed and schedule, the modes shared the exact records and update indices. The 1,173 authored-duration swaps were reserved for evaluation. The held-out generated tests used 512 new examples per shape and seed.

The numeric transform divides each duration by the plan mean, expresses ordinal distance as a fraction of plan length, and scales elapsed time by the exact suffix duration from the anchor. The position transform maps every sequence to the same 0–32 coordinate span and interpolates the relative lag bias. The suffix-mean control receives the same exact suffix total as the normalized model and assumes equal durations across that suffix. [Training runner](../../scripts/run_kam_duration_coordinates.py) · [Suffix control](../../scripts/replay_kam_duration_suffix_mean.py)

| Held-out events, maximum duration 90 | Short raw | Short numeric scaled | Short position scaled | Short both scaled | Wide raw | Wide both scaled | Suffix mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 87.57% | 87.70% | 86.20% | 90.89% | 75.72% | 89.19% | 53.32% |
| 32 | 79.43% | 77.02% | 76.56% | 78.97% | 66.73% | 80.08% | 41.54% |
| 64 | 13.80% | 6.45% | 15.36% | 51.30% | 46.61% | 66.15% | 33.98% |
| 128 | 1.11% | 3.32% | 3.71% | 22.07% | 26.56% | 44.47% | 26.63% |
| 256 | 0.91% | 2.93% | 2.73% | 8.40% | 4.43% | 23.63% | 20.12% |

The 256-event cases exceeded all training lengths. The wide normalized model beat suffix mean on two of three seed panels, recovering 134, 100, and 129 events versus 92, 114, and 103. At length 128 with shorter durations and roughly matched training span, wide normalization reached 46.74% and suffix mean 27.02%. The exact prefix control recovered all generated cases.

| Typed authored schedules | Original correct | Intervention correct | Both correct | Choice changed on sham |
| --- | ---: | ---: | ---: | ---: |
| Short raw | 69.37% | 70.99% | 51.27% | 1.11% |
| Short numeric scaled | 39.13% | 43.71% | 24.13% | 5.83% |
| Short position scaled | 59.79% | 59.87% | 37.88% | 2.50% |
| Short both scaled | 80.14% | 78.15% | 62.92% | 2.61% |
| Wide raw | 59.22% | 59.25% | 37.79% | 1.62% |
| Wide both scaled | 77.78% | 74.48% | 58.62% | 3.55% |
| Suffix mean | 34.19% | 24.04% | 0.00% | 0.00% |

Short normalization recovered both labels on 740, 674, and 800 authored pairs per seed versus 607, 535, and 662 with raw features. Typed anchor and elapsed time were supplied by the parser. HPG L4 array 52965583 generated the runs. [Seed-level outcomes](../../results/nonlinear_retrieval_natural/duration_coordinates_v1/) · [Earlier architecture comparison](DURATION_POINTER.md)
