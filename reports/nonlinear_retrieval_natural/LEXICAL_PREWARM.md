# Prewarming and threading exact Tantivy search

On the 128 HotpotQA timing questions, exact CPU lexical search fell from **150.64 ms to 58.22 ms** at the median after prewarming the shared Tantivy files and using two segment threads. The paired median saving was **92.06 ms** (95% question-bootstrap interval **75.66–103.62 ms**). All **16,384 ordered candidate IDs** matched across all three conditions. [Timing and parity audit](../../results/nonlinear_retrieval_natural/lexical_prewarm_v1/summary.json)

| CPU lexical condition | p50 | p95 |
| --- | ---: | ---: |
| Serial first pass | 150.64 ms | 298.49 ms |
| Serial after prewarm | 88.78 ms | 143.23 ms |
| Two segment threads after prewarm | 58.22 ms | 91.90 ms |

Prewarming alone saved **57.85 ms** paired median (**48.82–69.16 ms**). On warmed files, threading saved **30.01 ms** (**27.43–32.06 ms**) and was faster on 125 of 128 queries. Reading 380,242,985 bytes across 13 index files took 0.31–0.32 seconds per host. [Per-host records](../../results/nonlinear_retrieval_natural/lexical_prewarm_v1/)

HPG job **52958881** used two CPU hosts with two cores per worker and the same shared lexical index as the B200 hybrid path. Each host ran four separate warmup questions, measured 64 first-pass serial queries, prewarmed the files, and measured the other host's 64 questions. The warmed serial and two-thread requests ran in seeded within-question order. All paths used the original query parser, top-128 depth, and row-ID materialization. File-cache eviction was outside the protocol. [Driver](../../slurm/nonlinear_retrieval_lexical_prewarm.sbatch)

These results cover only exact BM25 up to the ordered list. The previous full B200 timing found 140.26 ms at the median for the initial hybrid search with ordinary lexical warmup. The CPU result identifies two changes to test at that entry point. End-to-end and concurrent serving performance need their own measurements. [Full B200 request timing](LIVE_DENSE_HYBRID_TIMING.md)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/lexical_prewarm_v1"
python scripts/analyze_kam_lexical_prewarm.py "$study"/prewarm-[01].json \
  --fixture "$base/dense_hybrid_anchor_v1/fixture.json.gz" --output /tmp/kam-lexical-prewarm.json
diff -u "$study/summary.json" /tmp/kam-lexical-prewarm.json
```
