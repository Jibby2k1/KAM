# Physical compaction after hard clock admission and address rank

On HPG L4, removing the unavailable suffix cut synchronized eager FP32 wall time by 6.2× for the radial decoder on a depth-1,024 mixed-future batch. At the same depth and batch width, standard histories saw a 4.1× reduction. All 2,304 paired episode decisions per compaction method matched the padded reference. The logits differed by up to 3.25 × 10⁻⁵.

Each geometry starts from three independently refit guarded, rank-augmented decoders. Sampled training losses and generated held-out decisions reproduce the archived runs exactly before the timing begins. All arms apply the same clock normalization, address-local rank and stable clock admission to the whole input. They then differ only in the physical suffix retained during causal encoding. The timing includes preprocessing and host synchronization.

At each depth and nuisance condition, 64 procedural episodes per fit are paired with the padded reference. Timing uses 20 synchronized repeats on one seed-rotated batch per fit. The table shows medians across the three fitted models and decimal MB of extra live allocations. A synthetic intervention changes publication time for some unrelated records and verifies that eligible target records remain identical.

| Geometry and synthetic depth | Batch | Condition | Padded wall | Batch-max wall | Length buckets wall | Padded → lowest peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Dot, 128 | 16 | Standard | 1.90 ms | 1.91 ms | 2.01 ms | 47.0 → 16.8 MB |
| Radial, 128 | 16 | Standard | 2.72 ms | 2.30 ms | 2.41 ms | 84.0 → 26.0 MB |
| Dot, 512 | 8 | Standard | 14.76 ms | 3.23 ms | 3.84 ms | 293.5 → 72.6 MB |
| Radial, 512 | 8 | Standard | 22.08 ms | 4.41 ms | 5.23 ms | 564.9 → 132.2 MB |
| Dot, 1,024 | 4 | Standard | 27.48 ms | 6.94 ms | 7.26 ms | 565.6 → 129.0 MB |
| Radial, 1,024 | 4 | Standard | 41.61 ms | 10.22 ms | 10.60 ms | 1,103.3 → 243.7 MB |
| Dot, 1,024 | 4 | Mixed future | 27.43 ms | 6.92 ms | 4.94 ms | 565.6 → 41.3 MB |
| Radial, 1,024 | 4 | Mixed future | 41.73 ms | 10.19 ms | 6.71 ms | 1,103.3 → 69.9 MB |

Batch-max was faster than length buckets at depth 512 under both profiles. At depth 1,024, length buckets paid off when eligible record counts varied from 439 to 944 across the timed batches. The standard profile retained 913 to 944. Measured preprocessing alone was 0.68–0.72 ms. These are preloaded GPU tests of elapsed compute and allocation. The comparison applies to these layouts and kernel dispatches, and does not establish the same speedups with a different attention implementation or request arrival process.

HPG L4 array 53011978. [Raw paired outcomes and timings](../../results/nonlinear_retrieval_natural/typed_guard_cropping_v1/) · [Hard admission and rank](TYPED_CLOCK_GUARD.md) · [Prior checkpoint compaction](PACKED_CLOCK_INFERENCE.md)
