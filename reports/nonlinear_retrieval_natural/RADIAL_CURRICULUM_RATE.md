# Learning-rate decay after a shared training checkpoint

At 19,200 matched updates, decaying the learning rate after update 9,600 increased joint accuracy on unseen 1,024-record histories from 44.14% to 50.91% for optimized dot and from 75.91% to 79.69% for optimized radial. The dot increase was consistent across all three seed-specific intervals. Radial remained ahead at this matched exposure.

Each geometry resumed the same model parameters and both AdamW moment buffers from its 9,600-update prefix. Prefix predictions matched the wall-clock study exactly on all held-out depths and seeds. Every branch consumed the same additional per-depth index sequence and 9,600 updates. One held the learning rate at 0.0004. The other used cosine decay from that rate toward 0.00004. The context kernels used the same short eager FP32 and long BF16 Flash settings, and evaluation used the same eager FP32 inference.

| Matched update count | Dot constant / cosine, joint 1,024 | Radial constant / cosine, joint 1,024 |
| ---: | ---: | ---: |
| 12,000 | 41.28% / 39.97% | 62.37% / 60.42% |
| 14,400 | 41.41% / 44.53% | 70.05% / 75.52% |
| 16,800 | 42.58% / 48.70% | 74.09% / 78.26% |
| 19,200 | 44.14% / 50.91% | 75.91% / 79.69% |

At 19,200, the dot gain ranged from +5.08 to +8.20 percentage points across the three seeds, with paired 95% intervals [+3.52, +12.89], [+1.95, +11.72], and [+0.39, +9.77]. Radial gained +2.73 to +4.69 points, with two intervals spanning zero. The late dot result shows sensitivity to optimization under repeated training draws. This experiment fixes update counts. The separate equal-time frontier uses the constant-rate schedule. It uses procedural data and an existing held-out fixture.

HPG L4 array 52985296 ran the three paired seeds. [Per-seed checkpoints and paired outcomes](../../results/nonlinear_retrieval_natural/curriculum_rate_v1/) · [Matched wall-clock comparison](RADIAL_CURRICULUM_WALLCLOCK.md)
