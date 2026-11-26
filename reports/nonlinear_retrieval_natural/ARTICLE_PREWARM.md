# Prewarming the shared article index on HPG

Sequentially reading the 1.66 GB shared article index before serving lowered median lookup time from **34.73 ms to 0.54 ms**. The paired median saving was **34.18 ms** (95% question bootstrap **33.69–35.14 ms**). The two prewarm passes took **0.74 and 1.02 seconds**. All 256 paired lookups returned identical ordered text.

| Phase | p50 | p95 |
| --- | ---: | ---: |
| Baseline first pass | 34.73 ms | 56.46 ms |
| After sequential prewarm | 0.54 ms | 0.69 ms |

The [storage experiment](ARTICLE_STAGING.md) previously compared two already warmed stores. Here, HPG job **52951154** used two distinct CPU nodes with one task per node. Each node read 128 requests from the existing frozen expansion-guard replay and waited for the other node's initial phase to finish. Each then read the full SQLite file sequentially and looked up the other node's 128 requests. The assignments ensure that each question appears in the two phases on different hosts. The workload includes the original pool and added conditioned articles, with order and retrieval choices fixed.

The benchmark measures SQLite candidate text fetch. It excludes tokenization, ranking, readers, and startup beyond the explicitly timed prewarm. The nodes share the underlying file server and baseline caches were not deliberately evicted. The results establish a large benefit for the controlled first-pass versus post-prewarm workload on these two nodes. Long-term readiness under concurrent load or memory pressure needs separate testing.

## Reproduce

Use the [HPG driver](../../slurm/nonlinear_retrieval_article_prewarm.sbatch) with the article index and archived expansion-guard replay. The [timing records and audit](../../results/nonlinear_retrieval_natural/article_prewarm_v1/) preserve assignments and ordered-text fingerprints.

```bash
base=results/nonlinear_retrieval_natural
study="$base/article_prewarm_v1"
guard="$base/expansion_guard_v1"
python scripts/analyze_kam_article_prewarm.py "$study"/prewarm-[01].json \
  --pools "$guard/pools.json.gz" --reference "$guard/replay.json.gz" --output /tmp/kam-prewarm.json
diff -u "$study/summary.json" /tmp/kam-prewarm.json
```
