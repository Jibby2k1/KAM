# Token-matched training with long correction histories

Mixing long histories into training reduced correction accuracy at all three large test depths at a fixed token and update budget. Delaying long updates improved the cyclic mixed curriculum, with depth-1,024 accuracy rising from 1.30% to 7.23%. The matched short-only mixed fit reached 21.09%.

Training uses 3,200 updates and 10,556,800 total tokens in each arm. Short-only alternates depth 16 with batch 100 and depth 48 with batch 34. The long curriculum cycles depths 16, 48, 128 and 512 with batch sizes 82, 42, 13 and 3. The short-depth source episode banks are fixed. Sample counts, batch sizes and distribution of history depths change under the equal aggregate budget. In mixed training the short-depth batches include archive-shaped examples, so introducing deeper procedural batches also reduces the overall archive-shaped fraction. The correction-only comparison isolates long exposure from the archive mixture, with the corresponding change in per-depth sample allocation. Gold and distractor positions are randomized. Radial and dot fits use the position-free architecture and start from matched shared parameters.

| Signed-log radial training | Held-out corrections, 48 | Corrections, 512 | Corrections, 1,024 | Earlier Philadelphia, 48 | Dallas, 48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short correction-only | 65.43 | 28.45 | 22.01 | 0.05 | 0.00 |
| Cyclic long correction-only | 34.70 | 11.07 | 5.34 | 0.00 | 0.00 |
| Short mixed | 79.82 | 30.53 | 21.09 | 81.25 | 42.44 |
| Cyclic long mixed | 10.55 | 1.43 | 1.30 | 56.67 | 20.61 |
| Delayed long mixed | 28.65 | 9.90 | 7.23 | 58.85 | 21.40 |

Values are joint exact percentages. The order-only contrast holds every per-depth batch, its contents, optimizer schedule by update, source hashes and initialization fixed. It starts with 800 short updates, then reproducibly shuffles the remaining depth schedule while traversing each depth bank in order. Timing and tail ordering differ from the cyclic schedule. The intervention recovered 18.10 points at depth 48 and 8.47 points at 512 for signed-log radial, while archive differences remained uncertain. Earlier Philadelphia changed by 2.19 points [−11.56, 12.19], and Dallas by 0.80 [−7.03, 8.61]. Both protocol differences matter when interpreting improvements from long-context training.

HPG L4 arrays 52994344–52994347 and 52997770. [Schedules, per-seed comparisons and archived outcomes](../../results/nonlinear_retrieval_natural/temporal_long_exposure_v1/) · [Mixed clock setup](TEMPORAL_CLOCK_CURRICULUM.md)
