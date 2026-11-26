# Dense and hybrid starting pools for conditioned expansion

On 1,024 fresh HotpotQA bridge questions, hybrid search placed both gold documents in **659** initial pools, versus **603** with dense search. With broad-ranked conditioned expansion on both arms, hybrid recovered **543** gold pairs and **340** joint exact matches, versus **526** and **328** with dense search. The paired joint interval reached zero. The primary quality comparison used HPG B200s and fixed candidate depth.

## Matched protocol

Both starting pools came from the same 5.20 million article index. Dense used the top 64 exact dense hits. Hybrid fused the top 128 BM25 and 128 exact dense hits with reciprocal rank fusion (rank constant 60) and kept 64. Both ordered pools were sealed before reranking. The same learned broad reranker picked the initial pair. The broad-ranked expansion used the first document of that pair as its anchor, appended distinct results from an exact conditioned top-64 search, and reranked with the frozen broad policy. The frozen reader decoded the selected pair. For joint EM, the pair must contain both annotated article IDs and the answer must match exactly after normalization. The new fixture was disjoint from the 22 reserved fixtures and native evaluation set by question ID, normalized text, and gold title. [Fixture and provenance](../../results/nonlinear_retrieval_natural/dense_hybrid_anchor_v1/)

| Initial search and policy | Both gold in candidates | Gold pair shortlisted | Both supports selected | Joint EM |
| --- | ---: | ---: | ---: | ---: |
| Dense, broad | 603 | 566 | 456 | 284 |
| Hybrid, broad | 659 | 606 | 484 | 301 |
| Dense, broad-ranked expansion | 734 | 664 | 526 | 328 |
| Hybrid, broad-ranked expansion | 780 | 695 | 543 | 340 |

Hybrid versus dense gained **+2.73 percentage points** in support recovery (paired 95% interval **+0.88 to +4.59**) and **+1.66 points** in joint EM (**+0.10 to +3.13**) with broad only. After broad-ranked expansion, it gained **+1.66 points** in support (**+0.20 to +3.13**) and **+1.17 points** in joint EM (**0.00 to +2.34**). Of the 56 net additional initially complete pools, hybrid gained 83 and lost 27. Only 10 questions lacked both supports under both starting methods. [Paired audit](../../results/nonlinear_retrieval_natural/dense_hybrid_anchor_v1/summary.json)

## Effect of expansion on complete and incomplete pools

Each row uses its own method's original completeness. The dense and hybrid complete subsets contain different questions.

| Original pool | Questions | Support, broad → expansion | Joint EM, broad → expansion |
| --- | ---: | ---: | ---: |
| Dense complete | 603 | 456 → 445 | 284 → 277 |
| Dense incomplete | 421 | 0 → 81 | 0 → 51 |
| Hybrid complete | 659 | 484 → 474 | 301 → 296 |
| Hybrid incomplete | 365 | 0 → 69 | 0 → 44 |

Across all questions, applying broad-ranked expansion to the hybrid pool added 39 net joint matches (paired gain **+3.81 points**, interval **+2.44 to +5.27**). It made 121 incomplete hybrid pools contain both supports. All 121 started with exactly one support. Among the 27 questions lacking both under dense, broad-ranked expansion completed one with dense and four with hybrid, with one joint answer for each. The summary also reports comparisons on identical question subsets defined by either starting method. [Paired audit](../../results/nonlinear_retrieval_natural/dense_hybrid_anchor_v1/summary.json)

A separate affinity-anchor diagnostic yielded 508 support and 317 joint for dense, and 528 support and 332 joint for hybrid. Dense affinity ran on B200, and hybrid affinity ran on L4. The original broad arm independently reproduced its selections, shortlist labels, and reader exactness on all 1,024 questions for each pool. The matched primary comparison takes broad and expanded outputs directly from the B200 broad-ranked jobs.

## Reproduce

HPG job **52950969** froze both pools. Arrays **52951608** and **52951609** scored the primary B200 comparison. The auxiliary affinity jobs were **52951122** and **52951123**. The quality replays used anchors verified against the first output of the frozen broad policy. They provide a matched-depth quality comparison. Live selection and initial search latency require separate measurements. [Pool freeze driver](../../slurm/nonlinear_retrieval_dense_hybrid_freeze.sbatch) · [Scoring driver](../../slurm/nonlinear_retrieval_dense_hybrid_anchor.sbatch)

```bash
study=results/nonlinear_retrieval_natural/dense_hybrid_anchor_v1
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_dense_hybrid_anchor.py "$study" \
  --output /tmp/kam-dense-hybrid-anchor.json
diff -u "$study/summary.json" /tmp/kam-dense-hybrid-anchor.json
```
