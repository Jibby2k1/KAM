# Conditioned expansion across three frozen HotpotQA cohorts

Across 3,072 disjoint bridge questions, the affinity-anchored union recovered **132 additional support pairs** and **104 additional joint exact matches** over broad reranking. All three cohorts improved overall. Each lost some joint matches in initially complete pools, and the union never recovered both supports when the original pool lacked both.

## Matched comparison

We pooled three archived evaluations retrospectively. The analyzer verifies disjoint IDs, normalized questions, and gold titles across all 3,072 queries, and matching index, model checkpoints, and reranker configuration across all shards. Within each question, broad ranks the original 64 articles. The conditioned arm appends the distinct hits from an exact top-64 corpus search conditioned on the original pool's affinity anchor, then reranks the union. We use the affinity run of the third cohort to keep the anchor policy fixed. Joint exact match requires the reader to answer exactly and both selected article IDs to match the gold evidence. [Conditioned union](CONDITIONED_UNION.md) · [Expansion guard](EXPANSION_REPLACEMENT_GUARD.md) · [Anchor comparison](BROAD_RANKED_ANCHOR.md)

| Cohort | Support pairs, broad → union | Joint EM, broad → union |
| --- | ---: | ---: |
| Conditioned union | 481 → 511 | 308 → 333 |
| Expansion guard | 464 → 512 | 297 → 341 |
| Anchor comparison, affinity arm | 487 → 541 | 321 → 356 |
| Combined | **1,432 → 1,564** | **926 → 1,030** |

The pooled support gain was **+4.30 percentage points** with a paired 95% interval of **+3.35 to +5.24**, from 179 wins and 47 losses. Joint EM rose **+3.39 points** with an interval of **+2.60 to +4.17**, from 130 wins and 26 losses. The intervals use 5,000 paired bootstrap resamples stratified by cohort. They describe variation across questions within these three bridge cohorts. [Audited counts and intervals](../../results/nonlinear_retrieval_natural/expansion_replications_v1/summary.json)

## Where the gains came from

| Original pool | Questions | Full coverage, broad → union | Gold pair shortlisted | Both supports selected | Joint EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete | 1,850 | 1,850 → 1,850 | 1,773 → 1,737 | 1,432 → 1,391 | 926 → 904 |
| One support missing | 1,114 | 0 → 280 | 0 → 245 | 0 → 173 | 0 → 126 |
| Both supports missing | 108 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

In initially complete pools, expansion displaced the gold pair from the shortlist on 36 questions and reduced joint EM by **1.19 points** (paired interval **−1.78 to −0.65**). In the initially incomplete pools, 173 of the 280 newly covered questions recovered both supports, and 126 also answered exactly. Of the 1,114 questions missing one support, the affinity anchor chose the available gold article on 449. The union completed 220 of these pools (49.0%) and 60 of the remaining 665 one-missing pools (9.0%). Anchor correctness is available for diagnosis with gold annotations. These data cover HotpotQA bridge questions evaluated with one frozen retriever, reranker, reader, and index. [Audited strata](../../results/nonlinear_retrieval_natural/expansion_replications_v1/summary.json)

## Reproduce the aggregation

```bash
base=results/nonlinear_retrieval_natural
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_expansion_replications.py "$base" \
  --output /tmp/kam-expansion-replications.json
diff -u "$base/expansion_replications_v1/summary.json" /tmp/kam-expansion-replications.json
```

The script reads the sealed fixtures, original ordered pools, and query-level replays from the three studies. GPU inference is unnecessary for the aggregation.
