# Radial training order at matched per-length exposure

Starting with shorter synthetic histories improved the held-out 1,024-record joint score from 48.70% to 52.47% after 9,600 updates. The longer unseen evaluation improved in all three seeds. The main effect appeared earlier in training and was larger at shorter training horizons. Short and intermediate-depth final scores were similar across curricula.

Both policies started from the same randomly initialized radial decoder with query-relative clocks and no positional encodings. Both consumed the same per-depth update indices in the same order within each depth. A shared 64-update warmup was followed by either interleaving all remaining updates or delaying long histories until each of the two short lengths had received half of its exposure. Both saw 2,400 updates at each depth of 16, 48, 128, and 512 records per bank and processed 39,307,200 tokens by the end. The depth-selective kernel, constant learning rate after warmup, optimizer, and precision were matched. Evaluation used a common eager FP32 path and fresh simulated histories.

| Held-out records per bank | Interleaved at 4,800 | Short first at 4,800 | Interleaved at 9,600 | Short first at 9,600 |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 83.46% | 83.46% | 94.40% | 93.36% |
| 128 | 50.33% | 58.20% | 85.03% | 84.44% |
| 512 | 22.40% | 30.27% | 61.85% | 63.48% |
| 1,024 | 11.59% | 19.53% | 48.70% | 52.47% |

At update 2,400, the short-first arm had never trained on 128- or 512-record histories. Its held-out 128-record joint score was 21.55% versus 1.30% for interleaving. By update 9,600 the 1,024-record gain ranged from +1.95 to +7.03 percentage points by seed. Final gains at 512 records were +3.52, +2.34, and −0.98 points. The 1,024-record histories were twice the maximum seen at training. Intermediate checkpoints align by update count and have slightly different token counts. Final counts and per-depth sample exposures are identical.

Update-only time ranged from 51.6 to 53.8 seconds per arm on L4. The update timer excludes data generation, checkpoint evaluation and logging. This is a synthetic schedule comparison with one model family and matched data. It isolates ordering from the source shift and length-exposure changes in the earlier [mixed long curriculum](RADIAL_MIXED_LONG_TRAINING.md). HPG array 52977654 executed all three seeds. [Trajectories and paired outcomes](../../results/nonlinear_retrieval_natural/radial_curriculum_order_v1/) · [Training runner](../../scripts/run_kam_radial_curriculum_order.py)
