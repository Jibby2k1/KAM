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

## Terminal-row readout with different cutoff positions

Computing only the query row in layer two reduced the standard depth-1,024 radial batch-max path from 10.03 to 5.65 ms. The same arm reached 5.65 ms with heterogeneous future arrivals. All 2,304 comparisons per terminal arm matched the full eager FP32 decisions. The largest absolute logit difference was 3.10 × 10⁻⁵.

A separate L4 replay crosses each of the three physical layouts with full attention or terminal-only attention in the last layer. The first layer is unchanged. The terminal row is gathered at each request's own cutoff, and its later keys are masked. Its residual, feedforward and output projection are evaluated at that row. Each run reconstructs the same eligible prefixes, and all first-layer states still precede any terminal intervention. Paired timings below come from this replay.

| Depth 1,024 at width four | Padded eager | Padded terminal | Batch-max eager | Batch-max terminal | Bucket eager | Bucket terminal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dot, standard | 27.43 ms | 14.33 ms | 6.81 ms | 3.94 ms | 7.17 ms | 4.15 ms |
| Dot, mixed future | 27.39 ms | 14.25 ms | 6.81 ms | 3.95 ms | 4.33 ms | 4.08 ms |
| Radial, standard | 41.58 ms | 21.48 ms | 10.03 ms | 5.65 ms | 10.48 ms | 5.91 ms |
| Radial, mixed future | 41.64 ms | 21.52 ms | 10.02 ms | 5.65 ms | 5.87 ms | 5.43 ms |

The standard radial batch-max path reached the same 243.7 MB extra live allocation with and without terminal selection. The mixed-future bucket path reached 69.9 MB in both variants. A whole-layer retained prefix still determines the activation peak. At the shallow standard depth, the shape changes and terminal readout gave similar small savings or overhead, depending on geometry.

HPG L4 array 53013166. [Combined readout and cropping measurements](../../results/nonlinear_retrieval_natural/typed_guard_terminal_v1/)
