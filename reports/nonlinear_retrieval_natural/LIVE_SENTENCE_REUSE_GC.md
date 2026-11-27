# Python collection pauses in sentence-reuse serving

A B200 repeat traced **one generation-2 Python garbage collection pause per shard**, each on a sentence-reuse request. The four pauses took **83.05–113.05 ms** and lined up with the same request positions that spiked in the earlier benchmark. In each trace the remaining-rerank stage was the only timed stage long enough to contain the recorded pause. [Trace and stage audit](../../results/nonlinear_retrieval_natural/live_anchor_reuse_gc_v1/summary.json) · [Original benchmark](LIVE_ANCHOR_SENTENCE_REUSE.md)

| B200 shard | Paired request index | Generation-2 pause | Remaining rerank |
| --- | ---: | ---: | ---: |
| 0 | 6 | 95.51 ms | 113.46 ms |
| 1 | 7 | 113.05 ms | 133.98 ms |
| 2 | 7 | 99.43 ms | 125.97 ms |
| 3 | 9 | 83.05 ms | 97.21 ms |

The trace reproduced a **6.52 ms** paired median saving for sentence reuse (95% interval **5.94–7.00 ms**). The measured p95 was **119.49 ms** for separate encoding and **123.68 ms** for reuse. Subtracting traced collection time from each request gives diagnostic p95 values of 119.17 and 112.68 ms. Those arithmetic residuals are a way to inspect the trace and provide no measured serving speed after a collector change. [Paired audit](../../results/nonlinear_retrieval_natural/live_anchor_reuse_gc_v1/summary.json)

All 128 pairs preserved their sealed original pools, anchors, conditioned candidate order, shortlist labels, selected evidence, and reader exactness. The 14 stored guard-feature values were identical on 126 of 128 queries. B200 array **52955880** used the same broad-ranked warm-request crossover design and captured Python collection callbacks for both policies. All four generation-2 events occurred when sentence reuse ran second. Since the policies shared one process per shard and collection depends on earlier allocations, the trace identifies the direct pause in the observed requests. Placement and scheduling of collection under other process layouts still need an intervention test. [Driver](../../slurm/nonlinear_retrieval_live_anchor_profile.sbatch)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_anchor_reuse_gc_v1"
ref="$base/broad_anchor_v1"
python scripts/analyze_kam_live_gc_trace.py "$study"/profile-52955880-*.json.gz \
  --fixture "$ref/fixture.json.gz" --reference "$ref/replay.json.gz" \
  --output /tmp/kam-live-gc.json
diff -u "$study/summary.json" /tmp/kam-live-gc.json
```
