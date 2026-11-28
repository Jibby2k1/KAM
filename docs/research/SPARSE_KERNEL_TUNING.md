# Tune the protected sparse prefill kernel on HPG

Launch tuning reduced isolated attention CUDA time by 12.70–18.02% while preserving all checked outputs bitwise. Three of the four locked configurations use the single-launch causal router. The corresponding end-to-end prefill gain was 0.49–0.63% against the same protected attention policy.

## Matched calibration

Compare the rectangular router with one GPU launch that chooses a smaller proxy tile for earlier query programs. Cross both routers with the default sparse softmax and nine configurations of 2, 4 or 8 warps and 1, 2 or 3 pipeline stages. A candidate must return exactly the original route indices and BF16 output values before timing. Lock a separate configuration at batch one and four and context 4K and 8K, requiring at least a 1% synthetic median CUDA gain. Then check the locked policies on frozen model logits and benchmark interleaved full-reader prefill on the RTX PRO 6000.

| Batch × context | Locked launch | Kernel CUDA p50 reduction | Protected p50 | Tuned p50 | Flash p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 × 4K | Rectangular, 4 warps, 2 stages | 12.70% | 185.81 ms | 184.69 ms | 194.39 ms |
| 1 × 8K | Single launch, 4 warps, 2 stages | 16.14% | 409.17 ms | 406.60 ms | 450.57 ms |
| 4 × 4K | Single launch, 4 warps, 2 stages | 17.76% | 807.40 ms | 803.45 ms | 834.17 ms |
| 4 × 8K | Single launch, 4 warps, 2 stages | 18.02% | 1,630.32 ms | 1,621.54 ms | 1,783.78 ms |

Eight of twenty configurations per shape qualified for timing. The locked configurations reproduced 16 evaluated A/B logits at each shape bitwise, with eight frozen prompts and both answer logits. Timing used 25 interleaved kernel trials per shape and 15 full-model trials. The Flash wall-time comparisons retain the protected policy's quality tradeoff. Compare the speedup between the two protected versions when evaluating the software change.

[Exact parity and raw timings](../../results/general_benchmarks_sparse_kernel_tuning/summary.json) · [HPG receipts](../../results/general_benchmarks_sparse_kernel_tuning/scheduler.tsv)
