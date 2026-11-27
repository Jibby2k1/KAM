# Optimized dot and radial learning at matched wall time

With CUDA Graph capture enabled for both geometries, dot learned faster at the first elapsed budgets and radial was ahead by the third checkpoint on L4 and B200. Across three B200 seeds at 24 seconds, median joint accuracy on 1,024-record histories reached 79.30% for radial and 47.27% for dot. The corresponding 60-second L4 result was 83.20% and 51.17%.

The two geometries start from identical shared parameters within each seed. They replay the same from-scratch short-first curriculum and learning-rate schedule, with the per-depth selective kernel for each geometry. A per-shape CUDA Graph captures each update, including AdamW. Time includes graph preparation and synchronized update blocks. It excludes data preparation and held-out evaluation. The first checkpoint at or just after each budget is used, polling every 48 updates. The 19,200-update control reproduced the earlier captured loss and all held-out flags for all 12 paired geometry and seed replays.

| GPU | Elapsed budget | Dot depth-1,024 joint | Radial depth-1,024 joint | Dot updates | Radial updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4 | 15 s | 33.59% | 10.55% | 6,672 | 4,512 |
| L4 | 30 s | 41.41% | 39.45% | 13,296 | 8,496 |
| L4 | 45 s | 47.66% | 71.48% | 20,112 | 12,672 |
| L4 | 60 s | 51.17% | 83.20% | 26,928 | 16,944 |
| B200 | 6 s | 32.03% | 5.86% | 5,664 | 3,792 |
| B200 | 12 s | 37.50% | 32.81% | 12,000 | 7,536 |
| B200 | 18 s | 45.31% | 62.11% | 18,384 | 11,328 |
| B200 | 24 s | 47.27% | 79.30% | 24,768 | 15,168 |

Each column reports the median across seeds. At the third and fourth checkpoint, every seed's paired radial-minus-dot interval excluded zero on both GPUs. The slowest crossing of a fixed elapsed budget was under 0.18 seconds past the target on L4 and under 0.10 seconds on B200. Dot processed more updates throughout. Measured accuracy and the crossover apply to this position-free synthetic task, the fixed short-first schedule and the selected kernels.

HPG arrays 52989645 and 52989646. [Joined measurements](../../results/nonlinear_retrieval_natural/captured_wallclock_v1/) · [Graph parity controls](RADIAL_CURRICULUM_CAPTURE.md) · [Earlier noncaptured timing](RADIAL_CURRICULUM_WALLCLOCK.md)
