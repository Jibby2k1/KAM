# L4 queue-inclusive batching at matched arrivals

At roughly 29.5 offered requests per second, allowing FP16 batches up to 16 completed 27.75 requests per second, compared with 26.43 for a cap of eight. Median shard p95 end-to-end latency fell from 1,345 to 1,128 ms. At roughly 23.6 offered requests per second, the cap of eight yielded lower p95 with a slightly lower completion rate.

One worker serves each of four disjoint shards on L4. Each rate and policy uses the same 384 scheduled Poisson arrivals and query draws per shard, with independent arrival seeds across shards and random crossover order. Workers use a FIFO queue and a five millisecond collection timeout. They launch the largest available supported batch in 1, 2, 4, 8 and 16, bounded by the arm's cap. Every arm uses the same FP16 sentence encoder, shared reranker features and prompt passage caching, and FP32 answer generation. The INT8 arm changes only the first-pass shortlist. The timer starts at actual in-process arrival and ends at CUDA-synchronized completion. Actual arrival jitter was small at all three offered loads, with median shard p95 below 1.7 ms.

| Approximate offered requests/s | FP16 cap 8, completed/s / p95 ms | FP16 cap 16 | INT8 cap 16 |
| --- | ---: | ---: | ---: |
| 11.80 | 11.79 / 227 | 11.79 / 232 | 11.79 / 220 |
| 23.59 | 22.59 / 542 | 23.25 / 649 | 23.15 / 567 |
| 29.49 | 26.43 / 1,345 | 27.75 / 1,128 | 27.15 / 1,195 |

Rates and p95 values are medians across four measured traces. At each offered rate all policies select identical ordered pairs and return identical support, answer, and joint flags on all 1,536 requests. Each rate includes repeat draws within shards. The offered rate exceeds every observed completion rate at the highest setting, and the width-16 FP16 median drain time at the end of that trace is 718 ms, versus 1,156 at width eight. With cap 16, the INT8 p95 advantage was confined to the two lighter workloads. This is a finite one-worker trace with in-process arrivals. It quantifies queue effects alongside the independent preloaded tests.

HPG L4 array 52999961. [Timed FIFO traces and summaries](../../results/nonlinear_retrieval_natural/fullwiki_live_queue_v3/) · [Static width and INT8 crossover](FULLWIKI_L4_THROUGHPUT.md)
