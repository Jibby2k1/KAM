# Sharing exact relation features across broad and expanded reranking

On the 1,024 held-out expansion-guard questions, computing relations once on the conditioned union and taking the original-pool subset reduced median feature time from **12.68 to 7.51 ms** (**1.69×**) and p95 from **16.08 to 9.60 ms**. Across both scorers, the saved GPU run reproduced every chosen evidence pair, gold-pair shortlist membership, and reader exactness label.

## Method and checks

The original candidate list is a prefix of the deduplicated conditioned union. Title graph edges between two original articles depend only on those articles. The paired lexical features depend on their text and the unchanged query. Compute the graph and pair features on the union with integer bitsets, then project its graph to the original prefix and take the original pair rows in their original triangular order.

For every question, both independently computed relation tensors and both projected tensors matched bitwise. The existing GPU scorer then consumed the shared relation features. Comparing against the frozen archived replay yielded 1,024 matches in each field for ordered pools, conditioning anchor, corpus hits, guard features, selected pairs, gold-pair shortlist membership, and exactness. The two scorers still form their own shortlists and normalize their scores within their own candidate sets.

HPG array **52949567** ran four B200 shards. The two CPU relation paths alternated order within each shard over identical article text and cached pair indices. The p50 and p95 aggregate all 1,024 questions. Timing covers feature calculation and projection. It excludes loading, candidate expansion, GPU reranking, and reader generation. The shared computation remains an opt-in verification path in this experiment.

## Reproduce

The [Slurm driver](../../slurm/nonlinear_retrieval_shared_relation.sbatch) uses the original expansion-guard fixture, sealed pools and reranker weights. The [replay and timing audit](../../results/nonlinear_retrieval_natural/shared_relation_v1/) include per-question measurements and the reference replay hash.

```bash
base=results/nonlinear_retrieval_natural
study="$base/shared_relation_v1"
python scripts/analyze_kam_shared_relation.py "$study/replay.json.gz" \
  --reference "$base/expansion_guard_v1/replay.json.gz" --output /tmp/kam-shared-relation.json
diff -u "$study/summary.json" /tmp/kam-shared-relation.json
```
