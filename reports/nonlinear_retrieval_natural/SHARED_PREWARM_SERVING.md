# Shared-index prewarming in B200 retrieval requests

On two HPG B200 nodes, sequentially prewarming the shared 1.66 GB SQLite article index reduced the median warm-model request from **132.34 ms to 63.29 ms** and p95 from **203.50 ms to 91.50 ms**. All 256 pairs returned the same candidate lists, relation features, evidence selections, and answer exactness. The prewarm took **0.59 and 0.55 seconds** on the two nodes.

| HPG host | First-pass p50 / p95 | Post-prewarm p50 / p95 |
| --- | ---: | ---: |
| Host 0 | 167.37 / 218.47 ms | 61.39 / 95.96 ms |
| Host 1 | 102.98 / 135.29 ms | 64.62 / 85.27 ms |
| Combined | **132.34 / 203.50 ms** | **63.29 / 91.50 ms** |

The ratio of medians was **2.09×**. The median paired saving was **60.57 ms** (paired question bootstrap 95% interval **48.23–76.53 ms**). All 256 completed faster after prewarming. Both phases agreed with the batched archive on selected evidence and exactness for every question. The intervals reflect variation across the sampled queries on these hosts. [Timing records and audit](../../results/nonlinear_retrieval_natural/shared_prewarm_serving_v1/)

## Request and crossover

HPG job **52951600** ran one B200 process on each of two separate nodes. Each process loaded the frozen index, reranker, and reader, then served four warmup questions outside the sample. Each timed 128 first-pass questions, waited for the other process, read the shared SQLite file sequentially, and timed the other process's 128 questions. This crosses question assignments over hosts while keeping the prewarm after the first pass on both. Both phases used affinity anchoring and exact shared relation extraction. The initial top-64 candidate pools were frozen. [Driver](../../slurm/nonlinear_retrieval_shared_prewarm_serving.sbatch) · [Prior CPU lookup experiment](ARTICLE_PREWARM.md) · [Frozen comparison](EXPANSION_REPLACEMENT_GUARD.md)

Each request starts at the sealed original pool, fetches original article text, encodes and conditions the query, searches the corpus for expansion, fetches additional articles, reranks both arms, and runs the reader before GPU synchronization. Timing excludes the original corpus search, model startup, and the separate readiness prewarm itself. Baseline caches were left in their first-pass state following the warmup. The benchmark measured single requests on these hosts in a fixed phase order. It leaves concurrent serving and the latency of broad-ranked anchoring open.

## Reproduce the audit

```bash
base=results/nonlinear_retrieval_natural
study="$base/shared_prewarm_serving_v1"
guard="$base/expansion_guard_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_shared_prewarm_serving.py "$study"/serving-[01].json.gz \
  --fixture "$guard/fixture.json.gz" --reference "$guard/replay.json.gz" \
  --output /tmp/kam-serving-prewarm.json
diff -u "$study/summary.json" /tmp/kam-serving-prewarm.json
```
