# CUDA graph prefill at matched batch and context

Compare eager and CUDA graph versions of full Flash, the protected shortlist and the final-256 Flash window. Use the same frozen Llama checkpoint, fixed-shape unpadded batches and identical GPU-resident inputs. The timed path includes all layers and the A/B projection. Graph timing excludes host transfer and graph construction. Capture runs separately on a warm side stream. [CUDA graph semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)

The first smoke run hit a materialized causal mask under capture. The retry used an explicitly unpadded causal dispatch and matched ordinary eager output bit for bit before profiling. All 18 captures matched their eager output bit for bit. Replaying the same graphs after changing the GPU input buffer also matched exactly.

## HPG result

Wall p50 in milliseconds over twelve interleaved trials per arm and shape. Cells show eager / graph. RTX PRO 6000, PyTorch 2.8.0.

| Batch × input tokens | Flash | Protected shortlist | Final 256 |
| ---: | ---: | ---: | ---: |
| 1 × 4,096 | 194.30 / 193.81 | 185.49 / 184.69 | 188.23 / 186.58 |
| 1 × 8,192 | 446.75 / 445.95 | 405.85 / 404.80 | 410.77 / 410.20 |
| 2 × 4,096 | 421.89 / 421.20 | 407.58 / 406.52 | 411.83 / 410.83 |
| 2 × 8,192 | 889.98 / 889.46 | 811.66 / 811.16 | 820.66 / 819.77 |
| 4 × 4,096 | 835.88 / 835.87 | 810.27 / 810.00 | 817.78 / 817.24 |
| 4 × 8,192 | 1,774.98 / 1,773.96 | 1,622.51 / 1,622.66 | 1,638.69 / 1,639.08 |

The largest graph speedup was 0.88% for the final-window mode at batch one and 4K. The final-window advantage over Flash was 7.60% for the batch-four 8K graph, close to the 7.68% eager advantage. Capture took about 127–145 ms per shape and mode on this GPU. At batch four and 8K, the measured incremental PyTorch allocation peak during capture was about 3.63 GiB. The reported allocations omit graph-pool reservations.

The long-prefill timing is largely insensitive to replacing eager dispatch with captured launches at these shapes. Preserve the graph path as a verified software control. The independent transfer study gives the corresponding quality limitation for the faster final-window attention. [Independent transfer](QUERY_WINDOW_TRANSFER.md)

[Benchmark](../../scripts/benchmark_kam_query_graph.py) · [Summarizer](../../scripts/summarize_kam_query_graph.py) · [Raw timings](../../results/general_benchmarks_query_graph/graph-profile.json) · [Numeric summary](../../results/general_benchmarks_query_graph/summary.json) · [HPG jobs](../../results/general_benchmarks_query_graph/scheduler.tsv)
