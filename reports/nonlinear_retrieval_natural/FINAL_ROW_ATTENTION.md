# Readout-only and hard terminal attention

Computing only the final query row in the last layer halved eager FP32 time at depth 1,024 on L4. The unpruned BF16 Flash implementation was substantially faster than either eager path. Taking the hard top-one key in each final-layer head lost 95 joint outcomes of 3,504 on the Dallas vintage task, while top 16 lost 12.

In a causal two-layer decoder with position-free context attention and one final readout, the first layer still computes all rows. The second layer needs all keys and values but only the query at the readout position. Its subsequent residual, feedforward, normalization and projection are pointwise. The eager single-row variant computes the same final-row softmax using the original learned radial metric. The Flash variant keeps the augmented radial coordinates and computes the last query against all keys. Both kernels are compared with their own full two-layer reference. The hard variants use the same exact FP32 scores and keep only the top 1, 4, 16 or 64 keys per head in the second layer. They retain the dense first layer and score every remaining key in the second. This experiment isolates terminal hard selection.

| L4 depth-1,024 timing, batch 4 | Full eager FP32 | Readout-only FP32 | Full BF16 Flash | Readout-only BF16 Flash |
| --- | ---: | ---: | ---: | ---: |
| Prior procedural checkpoint | 40.74 ms | 20.62 ms | 1.55 ms | 1.56 ms |
| Reproduced mixed relative-clock fits | 40.74 ms | 20.63 ms | 1.42 ms | 1.45 ms |

Eager FP32 readout-only matched every alias and payload prediction across 37,056 comparisons over both models, four procedural lengths, two archives and two archived candidate depths. This includes repeated query symbolizations and seed replays. BF16 readout-only matched every archived decision and differed on eight of 9,216 procedural comparisons against the full BF16 reference. The FP32 speedup skips work in the second layer. Peak allocated memory at depth 1,024 was still about 1,045 MiB because the first dense layer has the same shape.

On Dallas at depth 48, the mixed relative-clock radial model reached 2,017 strict joint outcomes with full softmax and 1,922 with hard top one. The paired difference was −2.71 percentage points [−3.81, −1.68]. Top four reached 1,962 [−2.41, −0.68 points versus full], top 16 reached 2,005 with an interval crossing zero, and top 64 reached 2,013. The full readout returns the correct alias on 2,182 of 3,504 Dallas records and 1,392 of 3,456 Eurostat records. At that Eurostat depth it returns the exact snapshot on 126 and both fields on 80. On these typed inputs a deterministic latest-visible lookup given the model's alias can reach the alias-correct fraction. That diagnostic requires a typed record index and still inherits alias errors.

HPG L4 arrays 53001096 and 53001097. [Timing, paired outcomes and retained attention mass](../../results/nonlinear_retrieval_natural/final_row_attention_v2/) · [Physical clock compaction](PACKED_CLOCK_INFERENCE.md) · [Eurostat transfer](EUROSTAT_VINTAGES.md)
