# Compute only the causal prefix when building a block shortlist

The original Triton route builder performs a rectangular query–proxy product, then masks future columns. Its program handles 16 query tiles. Earlier programs can therefore compute a smaller power-of-two proxy tile and return the same routes. Compare the original rectangular implementation against tiled computation with the original head-dimension split, and tiled computation with that split chosen separately by proxy width. [Triton matrix multiplication and tiling](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

Match route indices and BF16 output values bitwise on synthetic grouped-query inputs including uneven lengths and batches one and four. Check the matched full-model A/B logits on eight frozen prompts per width. Profile shortlist construction, the whole attention kernel and frozen Llama prefill on HPG across 4K and 8K and batches one and four. The full-model profile includes both protected shortlisting and the selective final-256 Flash window.

## Results on HPG

Both tilings gave identical synthetic routes and BF16 attention outputs at all seven shapes and identical full-model A/B logits on all audited prompts. In theory, the tiling removed 32.8% of padded proxy dot-product work at 4K and 33.2% at 8K. It used four and five router launches respectively.

| Batch × context | Rectangular route CUDA p50 | Causal tiles route CUDA p50 | Rectangular protected model p50 | Causal tiles protected model p50 |
| --- | ---: | ---: | ---: | ---: |
| 1 × 4K | 38.66 µs | 72.02 µs | 189.37 ms | 189.67 ms |
| 1 × 8K | 49.47 µs | 91.38 µs | 411.66 ms | 411.75 ms |
| 4 × 4K | 69.12 µs | 81.46 µs | 815.27 ms | 814.83 ms |
| 4 × 8K | 121.34 µs | 115.46 µs | 1,638.63 ms | 1,638.25 ms |

At batch one, the entire sparse attention kernel also became slower, from 291 to 331 µs at 4K and 555 to 601 µs at 8K. For all full-model comparisons at the same attention policy, the absolute p50 change was under 0.17%. Incremental allocated peak was identical across policies within each shape. Keep the original rectangular route builder for the current configuration.

The mismatch between the theoretical dot-product reduction and the measured time motivates testing a single-launch implementation and tuning the retained-key softmax that dominates isolated sparse attention time. This experiment's parity checks covered frozen examples and randomized tensor shapes.

[Timing replay and parity](../../results/general_benchmarks_causal_proxy_tiles/summary.json) · [HPG receipts](../../results/general_benchmarks_causal_proxy_tiles/scheduler.tsv)
