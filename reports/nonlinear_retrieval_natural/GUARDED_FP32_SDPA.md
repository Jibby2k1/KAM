# Fused FP32 attention for guarded temporal lookup

The memory-efficient SDPA backend and batch-max cropping reduced synchronized L4 radial time from 39.28 to 1.79 ms on a depth-1,024 mixed-future batch of four. All three physical layouts preserved all tested FP32 alias and snapshot choices on L4, RTX PRO 6000, and B200. FP32 logits differed slightly. B200 radial was marginally faster at the padded shape, so the fastest layout depends on the device.

The three dot and radial checkpoints were deterministically refit on L4, reproducing the archived loss checkpoints and every measured generated outcome before saving the weights. Every device loads identical weights and the same CPU-generated episode histories. The clock guard, address-local rank, stable eligible prefix, and readout query position stay fixed. The new FP32 kernel uses PyTorch memory-efficient scaled dot-product attention. For radial scoring, the augmented queries and keys fold the key energy into an added coordinate, with the learned diagonal metric and bandwidth retained. Softmax already cancels the term that depends only on the query. BF16 Flash with augmented radial scores provides a precision comparison. The eager terminal variant computes only the second-layer query row.

| GPU and geometry, depth 1,024 mixed future | Padded eager FP32 | Batch-max terminal FP32 | Padded fused FP32 | Batch-max fused FP32 | Padded BF16 Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4, dot | 25.09 ms | 3.60 ms | 3.08 ms | 1.48 ms | 1.37 ms |
| L4, radial | 39.28 ms | 5.29 ms | 3.66 ms | 1.79 ms | 1.98 ms |
| RTX PRO 6000, dot | 5.01 ms | 1.23 ms | 1.24 ms | 1.04 ms | 1.08 ms |
| RTX PRO 6000, radial | 7.37 ms | 1.50 ms | 1.51 ms | 1.40 ms | 1.54 ms |
| B200, dot | 2.89 ms | 1.77 ms | 1.56 ms | 1.50 ms | 1.58 ms |
| B200, radial | 4.08 ms | 2.19 ms | 1.98 ms | 2.03 ms | 2.22 ms |

Per device and layout, fused FP32 agreed with the eager reference on 2,304 of 2,304 choices across the two geometries, three depths, two rejection profiles, and three independently trained seeds. The same scenarios repeat across layouts and devices. BF16 Flash agreed on 2,297 of 2,304 per device and layout in this study. At depth 1,024, radial batch-max fused FP32 used 19.7 extra allocated MB, down from 1,102.3 MB for padded eager and 243.7 MB for batch-max terminal. The table reports the median of three seed medians, each based on 20 synchronized repeats over a preloaded batch including the clock and rank preprocessing. This benchmark measures GPU-ready inputs and does not include queueing, data ingestion, search, or natural-language interpretation.

HPG arrays 53014085–53014088. [Per-example decisions and synchronized timing](../../results/nonlinear_retrieval_natural/typed_guarded_sdpa_v1/) · [Prior terminal-readout comparison](TYPED_GUARD_CROPPING.md)
