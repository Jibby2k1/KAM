# Moving Python collection into B200 worker readiness

An explicit generation-2 collection after warmup moved the recurring full-collection pauses outside the measured request window on four B200 workers. The normal processes had **four** generation-2 events across 256 requests. The two readiness variants had **zero** each and preserved every output. The intervention cost **87–142 ms** per worker for collection alone, and **84–161 ms** for collection plus freezing. [Timing, GC and parity audit](../../results/nonlinear_retrieval_natural/live_anchor_gc_readiness_v1/summary.json)

| Readiness policy | Full GC during requests | Separate sentence encoding p50 / p95 | Sentence reuse p50 / p95 |
| --- | ---: | ---: | ---: |
| Normal automatic collection | 4 | 84.12 / 124.65 ms | 75.27 / 116.78 ms |
| Full collection after warmup | 0 | 83.89 / 126.72 ms | 77.13 / 111.46 ms |
| Full collection then freeze existing objects | 0 | 82.52 / 117.78 ms | 76.48 / 107.11 ms |

The paired median sentence-reuse saving from freezing versus readiness collection alone was **0.13 ms** (95% question interval **−0.03 to +0.35 ms**). Both readiness policies slightly increased the median reuse request time relative to normal. The observed reuse p95 fell by 5.32 ms after collection and by 9.67 ms after collection plus freezing. The short sample supports moving the first full collection to readiness. It gives limited evidence about the incremental effect of freezing. [Per-request comparisons](../../results/nonlinear_retrieval_natural/live_anchor_gc_readiness_v1/)

## Matched worker processes

HPG array **52959337** ran three fresh processes on each of four GPUs with the readiness order varied across shards. Every process prewarmed the same article index and ran four disjoint warmup questions with and without sentence reuse. Each then timed the same 32 paired requests on its shard with the same within-question policy order. Automatic collection remained enabled across all conditions. Freezing placed roughly 369,500 already tracked objects in Python's permanent generation. [Prior collection trace](LIVE_SENTENCE_REUSE_GC.md) · [Python collector documentation](https://docs.python.org/3.11/library/gc.html) · [HPG driver](../../slurm/nonlinear_retrieval_gc_readiness.sbatch)

Both cached and uncached paths preserved ordered candidates, features, sentence counts, evidence selections and reader labels across readiness modes on all 128 matched questions. Peak host RSS after the request window stayed between **2.00 and 2.01 GiB**. That bound covers these short traced workers. A longer single-policy run is needed to tell whether collections stay cheaper after longer-lived request objects accumulate and to assess memory under load. [Trace artifacts](../../results/nonlinear_retrieval_natural/live_anchor_gc_readiness_v1/)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_anchor_gc_readiness_v1"
ref="$base/broad_anchor_v1"
python scripts/analyze_kam_gc_readiness.py "$study"/profile-52959337-*.json.gz \
  --fixture "$ref/fixture.json.gz" --reference "$ref/replay.json.gz" --output /tmp/kam-gc-readiness.json
diff -u "$study/summary.json" /tmp/kam-gc-readiness.json
```
