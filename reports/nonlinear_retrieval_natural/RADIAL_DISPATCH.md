# Depth-selective precision for radial updates

Keeping the short updates in eager FP32 and using balanced split BF16 FlashAttention at 1,025 tokens reduced L4 training wall time from 16.03 to 5.03 seconds. On B200 the same policy took 6.48 seconds versus 6.37 for eager FP32. At 512 records per bank, the joint accuracy differences from eager FP32 were −0.26 points on L4 and +0.13 points on B200.

## Checkpoint-matched comparison

All five modes began with the same pretrained radial decoder, which already solved 91.02% of the held-out 512-record panels. Three seeds on each GPU shared input bank contents, sampling indices, and evaluation examples across modes. The 1,200 updates cycled among 41, 257, and 1,025 tokens per example and processed 4,924,800 tokens in every arm. The dispatch policy used eager FP32 at 41 and 257 tokens and the balanced split BF16 Flash kernel at 1,025. The optimizer retained the same FP32 parameters throughout.

| Device and training mode | Full training path | Extra peak allocated GPU memory | Joint accuracy at depth 512 |
| --- | ---: | ---: | ---: |
| B200 eager FP32 | 6.37 s | 588 MB | 96.81% |
| B200 eager BF16 | 7.11 s | 611 MB | 97.20% |
| B200 original BF16 Flash | 6.68 s | 54 MB | 96.81% |
| B200 balanced split BF16 Flash | 6.58 s | 51 MB | 97.27% |
| B200 depth-selective | 6.48 s | 182 MB | 96.94% |
| L4 eager FP32 | 16.03 s | 588 MB | 96.94% |
| L4 eager BF16 | 14.72 s | 611 MB | 96.74% |
| L4 original BF16 Flash | 5.26 s | 54 MB | 97.01% |
| L4 balanced split BF16 Flash | 5.33 s | 51 MB | 96.35% |
| L4 depth-selective | 5.03 s | 182 MB | 96.68% |

Joint accuracy is the mean over 512 examples per seed, evaluated in the same eager FP32 path for all trained models. At depth 128 the selective path reached 98.11% on both devices. Eager FP32 reached 98.76% on B200 and 98.44% on L4. Both views and seed-level paired outcomes are recorded. These repeats assess the same initial checkpoint on different continuation streams.

The timing includes feature encoding, forward and backward, gradient clipping, dispatch, and the optimizer. Initial data generation and evaluation are outside the clock. Selective dispatch reduced the extra peak allocation by about 69% versus eager FP32 on both devices. The all-fused policies used less memory and ran within 0.30 seconds of selective dispatch on L4. The matched workload therefore supports a choice based on hardware and memory budget. HPG arrays 52969191 and 52969192 ran the comparison. [Runner](../../scripts/run_kam_radial_dispatch.py) · [Validated outcomes](../../results/nonlinear_retrieval_natural/radial_dispatch_v1/) · [Earlier from-scratch mixed curriculum](RADIAL_MIXED_LONG_TRAINING.md)
