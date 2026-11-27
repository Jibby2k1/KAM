# Radial attention training on B200 and L4

The augmented radial attention formulation preserved the FP32 task decisions and passed the specified gradient checks on both GPUs. At 1,025 tokens and batch 32, BF16 FlashAttention cut the training step from 18.73 to 6.08 ms on B200 and from 231.11 to 12.17 ms on L4. After a short matched continuation, the BF16 change improved depth-512 accuracy on B200 and reduced it on L4.

## Matched experiment

All modes began with one frozen two-layer, position-free radial decoder trained on 13,516,800 generated bitemporal tokens. Attention dropout was zero. Both GPUs used PyTorch 2.8.0 with CUDA 12.8. Inputs and checkpoint digests match across devices. The measured step included forward, label loss, backward, clipping, and fused AdamW, with five warm updates and 24 CUDA event measurements per shape.

| GPU | Records per stream | Input tokens | Batch | Eager BF16 p50 | Flash BF16 p50 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B200 | 20 | 41 | 32 | 6.38 ms | 5.99 ms | 1.07× |
| B200 | 128 | 257 | 32 | 6.25 ms | 6.01 ms | 1.04× |
| B200 | 512 | 1,025 | 32 | 18.73 ms | 6.08 ms | 3.08× |
| L4 | 20 | 41 | 32 | 4.02 ms | 3.86 ms | 1.04× |
| L4 | 128 | 257 | 32 | 13.34 ms | 3.90 ms | 3.42× |
| L4 | 512 | 1,025 | 32 | 231.11 ms | 12.17 ms | 18.99× |

At the largest shape, extra peak allocated CUDA memory during the measured step fell from 4,805 to 330 decimal MB on both GPUs. The timing includes potential host launch gaps in the measured stream. It excludes fixture generation and serving costs. The [raw records](../../results/nonlinear_retrieval_natural/radial_backward_v1/) give all three batch sizes and memory measurements.

## Gradients and continuation

Against eager FP32, the math backend had maximum absolute logit error of 0.000050 and relative L2 gradient error at most 0.000026 across three depths on both GPUs. All cases passed the predeclared logit tolerance with absolute and relative thresholds of 0.0002 and relative L2 gradient threshold of 0.001. The FP32 arms made identical held-out predictions on all 1,536 examples per device. Relative gradient error for BF16 Flash versus eager BF16 reached 0.367 at depth 128 on both devices.

Continuation used the same 128 updates and 305,152 tokens per mode, alternating 20 and 128 records per stream. The comparison below uses joint entity and value accuracy on independent 512-example sets at each depth.

| GPU | Depth | Eager FP32 | Augmented math FP32 | Eager BF16 | Augmented Flash BF16 | Flash vs eager BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B200 | 20 | 510 | 510 | 511 | 512 | +0.20 pp |
| B200 | 128 | 495 | 495 | 499 | 502 | +0.59 pp |
| B200 | 512 | 451 | 451 | 459 | 476 | +3.32 pp |
| L4 | 20 | 510 | 510 | 511 | 510 | −0.20 pp |
| L4 | 128 | 495 | 495 | 499 | 489 | −1.95 pp |
| L4 | 512 | 451 | 451 | 465 | 441 | −4.69 pp |

At depth 512 the paired question-bootstrap intervals were +1.56 to +5.08 points on B200 and −6.84 to −2.73 on L4. Those intervals condition on one checkpoint and one continuation per hardware and mode. They do not measure variation across training seeds. Across the continuation itself, Flash BF16 spent 0.79 seconds versus eager BF16's 0.81 seconds on B200 and 0.50 versus 0.53 seconds on L4.

The next study varies BF16 feature preparation and splits the key bias across two lanes at the same padded width. It uses three fresh continuation seeds to test whether a more precise encoding reduces gradient error and improves held-out accuracy consistently. HPG jobs 52960411 and 52960412 produced the present result. [Runner](../../scripts/benchmark_kam_radial_backward.py) · [Audit](../../scripts/analyze_kam_radial_backward.py) · [PyTorch SDPA backend documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)
