# Broad-ranked anchors improve conditioned corpus expansion

On 1,024 new frozen HotpotQA questions, the broad-ranked anchor recovered **560 support pairs and 369 joint exact matches**, compared with **541 and 356** for the previous affinity anchor. The change made **131 of 420 incomplete pools** contain both supports, versus **105** with the affinity anchor.

## Design

We reserved a new bridge-question cohort with no ID, normalized-text, or gold-title overlap with 21 archived fixtures. The ordered original 64 candidates were sealed before evaluation. The affinity anchor maximizes adapted query affinity among the original candidates. The broad-ranked anchor is the first document of the original broad reranker's predicted pair. It remains fixed even if a different pair wins after expansion.

Both expansion arms use a single anchor, the same query text template and frozen encoder and adapter, the exact fully indexed 64-document conditioned top list, and the same expanded reranker and reader. They average the same adapted original and conditioned query vectors. They exclude their anchor and add distinct selected items to the original 64. The average number of unique additions was 48.49 with affinity and 48.19 with broad-ranked anchoring. Broad-only reranking supplies a no-expansion baseline. Support recovery requires both gold document IDs, and joint EM additionally requires the reader's normalized exact answer match.

## Results

| Original pool | Questions | Broad only | Affinity-anchor union | Broad-anchor union |
| --- | ---: | ---: | ---: | ---: |
| All | 1,024 | 487 / 321 | 541 / 356 | **560 / 369** |
| Complete | 604 | 487 / 321 | 470 / 311 | **471 / 312** |
| Incomplete | 420 | 0 / 0 | 71 / 45 | **89 / 57** |
| One missing | 383 | 0 / 0 | 71 / 45 | **89 / 57** |
| Two missing | 37 | 0 / 0 | 0 / 0 | 0 / 0 |

Cells report recovered support pairs and joint exact matches. Relative to the affinity-anchor union, the broad-ranked anchor gained **1.86 percentage points** in support (paired 95% interval **+0.88 to +2.93**) and **1.27 points** in joint EM (**+0.49 to +2.15**). The incomplete slice gained **4.29 points** in support and **2.86 points** in joint EM. The complete slice gained one joint answer over affinity anchoring and remained nine below broad-only.

The broad-ranked anchor was itself gold evidence on 226 of the 383 one-missing cases, compared with 158 for affinity. Across all 398 questions where the anchors differed, the broad anchor was gold 249 times and affinity 16 times. Relative to affinity, improved coverage on incomplete pools translated to 26 more pools with both supports, 21 more with the gold pair shortlisted, 18 more pairs correctly selected, and 12 more joint matches. Neither anchor recovered both supports in the 37 two-missing cases.

The broad anchor now has a held-out accuracy advantage within this candidate budget. It requires the original reranker before the conditioned search. The phase order changes from affinity anchoring, so request latency needs a separate measurement.

## Reproduce

Array **52949717** froze the pools. HPG arrays **52950036** and **52950215** scored affinity and broad-ranked anchoring. Their independent original broad rankings, shortlist labels, and reader results matched on 1,024 out of 1,024 questions. The [Slurm driver](../../slurm/nonlinear_retrieval_broad_anchor.sbatch) runs the sealed comparison. The [study artifacts](../../results/nonlinear_retrieval_natural/broad_anchor_v1/) include the original pools, the affinity reference used to define broad anchors, and both expansion replays.

```bash
study=results/nonlinear_retrieval_natural/broad_anchor_v1
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_broad_anchor.py "$study/replay.json.gz" \
  --affinity "$study/affinity-replay.json.gz" --fixture "$study/fixture.json.gz" --pools "$study/pools.json.gz" \
  --output /tmp/kam-broad-anchor.json
diff -u "$study/summary.json" /tmp/kam-broad-anchor.json
```
