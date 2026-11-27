# Ordering the same training episodes under CUDA Graphs

A short-first order improved early depth-1,024 accuracy in the captured radial decoder on both devices. At 9,600 matched updates it reached 51.17% on B200 and 48.05% on L4, versus 43.36% and 45.31% under interleaving. By 19,200 updates the seed-level intervals for each schedule comparison included zero.

The comparison crosses radial and dot scoring with two orders for the same 38,400-update stream. It holds initialization, depth-specific minibatch sequence and contents, CUDA Graph implementation, and per-update learning rates fixed. The schedules differ only before step 9,600. The short-first arm advances through a curriculum with a smaller early depth. The interleaved arm spreads the same update counts across depths in that phase. Both have exactly the same token counts and updates at 9,600, followed by the same remaining batches. At 4,800, the depth allocation and token count can differ.

| Device and geometry | Depth-1,024 joint at step 4,800, interleaved / short-first | Step 9,600 | Step 19,200 |
| --- | ---: | ---: | ---: |
| L4 dot | 25.39% / 30.08% | 32.03% / 38.67% | 50.00% / 47.66% |
| L4 radial | 6.64% / 12.50% | 45.31% / 48.05% | 82.03% / 85.16% |
| B200 dot | 26.56% / 33.20% | 35.55% / 34.77% | 48.05% / 45.70% |
| B200 radial | 6.64% / 13.28% | 43.36% / 51.17% | 81.64% / 83.98% |

The table reports medians of three paired seeds on 256 held-out histories per seed. At 9,600 updates, all three B200 radial short-first intervals were positive. Two of the three L4 radial intervals were positive. Both devices reproduced all 12 earlier short-first checkpoints at 9,600 and 19,200 updates, including their recorded losses, held-out decisions, token counts, and depth allocation.

The elapsed-time comparison includes CUDA Graph setup and excludes time spent on checkpoint readout. The B200 radial median at 24 seconds was 79.30% short-first versus 76.95% interleaved. The L4 medians at 60 seconds were 83.20% and 82.03%. Earlier time checkpoints have greater relative gains for short-first. Dot had higher final medians under interleaving, with all paired 19,200-update intervals overlapping zero. The same training order can therefore change early convergence without producing a clear final gain at the tested budget.

HPG L4 array 52991926 and B200 array 52991927. [Per-seed outcomes and exact replay checks](../../results/nonlinear_retrieval_natural/captured_curriculum_order_v1/) · [Prior wall-clock comparison](RADIAL_CAPTURED_WALLCLOCK.md)
