# Collector readiness over 1,008 cached requests per mode

Four fresh B200 workers each encountered one 93–96 ms generation-2 pause under normal automatic collection. An explicit full collection at readiness moved that observed cost out of the request window. Freezing the existing objects after collection gave no consistent additional tail benefit.

## Matched worker replay

Each worker evaluated one disjoint 256-question shard of the frozen HotpotQA broad-anchor cohort. The first four cases warmed the worker, followed by 252 cached sentence-reuse requests in the original order. All three modes ran in fresh processes on the same hardware for each shard, with the mode order varied among shards. The timer starts with the initial frozen pool available and includes anchor reranking, conditioned search, reranking, and the reader. Only compact outcome fingerprints and timing data accumulate across requests. All modes kept automatic garbage collection enabled. [Earlier short crossover](LIVE_GC_READINESS.md) · [Raw records](../../results/nonlinear_retrieval_natural/live_anchor_gc_sustained_v1/)

| Readiness | Requests | Generation-2 pauses | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 1,008 | 4 | 79.06 ms | 114.04 ms | 159.91 ms |
| Full collection | 1,008 | 0 | 78.38 ms | 111.53 ms | 153.75 ms |
| Full collection, then freeze | 1,008 | 0 | 79.71 ms | 113.61 ms | 153.36 ms |

The four normal-mode pauses occurred at requests 17, 22, 21, and 20. Full collection took 87–96 ms at worker readiness. Relative to normal collection, per-worker p99 savings were +7.18, −0.35, +26.79, and +1.50 ms. Relative to collection alone, the savings from adding freeze were +4.22, −2.26, −11.41, and −0.34 ms. Resident memory rose by 77–85 MiB from readiness to the end across all modes, with similar peak CPU RSS near 2.07 GiB. These are four short-lived, serial replay workers on the same benchmark cohort.

Across all 1,008 requests, full outcome fingerprints matched across readiness policies. In the comparison with the batched archive, 1,007 matched the anchor, membership, pairs, shortlist labels and answer exactness directly. HPG job 52962277 reproduced the remaining request separately and found one substituted article in the conditioned set. Both selected pairs, shortlist labels and answer exactness still matched. The sole membership change occurred across microbatch shapes and was identical in all three readiness policies.

The timing array was HPG 52960822. The experiment supports pre-serving collection as a small startup cost in exchange for removing the first measured full pause. It provides no evidence of a further tail reduction from freezing across these worker-level flows. [Timing and fidelity audit](../../scripts/analyze_kam_gc_sustained.py)
