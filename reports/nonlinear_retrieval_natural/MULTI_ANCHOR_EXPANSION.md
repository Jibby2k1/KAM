# Multi-anchor HotpotQA expansion at a fixed candidate budget

On 1,024 fresh HotpotQA questions, four anchors completed 113 of 407 incomplete candidate pools, compared with 105 from one anchor. The additional coverage concentrated on questions where the top anchor was wrong. Sharing the 64 slots across anchors reduced coverage when the first anchor was already correct.

## Design

A new holdout of 1,024 bridge questions from the official HotpotQA training source was disjoint from 19 reserved fixtures by ID, normalized text, and gold article titles. A first HPG array froze the ordered 64-document original pools. All gold articles were present in the indexed corpus. Expansion used the same frozen query adapter and the same indexed Wikipedia snapshot for every method.

We used the most relevant one, two, or four original documents by adapted query affinity. Each anchor generated its own corpus search through the mean of the adapted original and anchor-conditioned query. We interleaved results by rank, skipped duplicates and original articles, and kept exactly **64 new documents**. For the unconditioned control, the adapted original query supplied the same 64-new-document budget. The two-anchor versus one-anchor comparison was primary. Four anchors and unconditioned search served as controls.

## Coverage

A pool is complete when it contains both gold support documents. The table counts pools made complete by the added documents. The complete frozen pools contribute the same 617 questions to every full-cohort total. Anchor correctness and support labels were used only for analysis.

| Originally incomplete cohort | Unconditioned | One anchor | Two anchors | Four anchors |
| --- | ---: | ---: | ---: | ---: |
| All 407 incomplete | 48 | 105 | 109 | **113** |
| Top anchor correct, 166 | 27 | **87** | 82 | 73 |
| Top anchor wrong, 241 | 21 | 18 | 27 | **40** |
| Top wrong and second correct, 41 | 7 | 5 | **14** | 13 |
| Both supports missing, 26 | 0 | 0 | 0 | 0 |

On all 407 incomplete pools, two anchors versus one gained **0.98 percentage points** (95% paired bootstrap interval **−1.47 to +3.44**). Four versus one gained **1.97 points** (**−0.98 to +5.16**). Within the top-anchor-wrong stratum, gains were **3.73 points** (**+0.41 to +7.05**) and **9.13 points** (**+5.39 to +13.28**), respectively. One anchor versus the unconditioned search gained **14.00 points** on incomplete pools (**+10.07 to +17.94**). Paired intervals use 5,000 resamples of questions.

The results suggest preserving the first anchor's candidate budget when its evidence is strong and spending slots on alternate anchors when it is weak. Anchor correctness in this report is a diagnostic label. A deployable policy would need an inference-visible estimate and a separate holdout. This experiment measured candidate availability. It did not rerank the expanded lists, run a reader, or benchmark serving latency.

## Reproduce

HPG array **52948191** froze the original pools. Array **52948562** ran the coverage comparison on four B200 GPUs. All four workers independently produced byte-identical sealed pool files. The [Slurm driver](../../slurm/nonlinear_retrieval_multi_anchor_expansion.sbatch), [holdout, pool provenance and paired replay](../../results/nonlinear_retrieval_natural/multi_anchor_expansion_v1/) record the inputs and results.

```bash
study=results/nonlinear_retrieval_natural/multi_anchor_expansion_v1
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_multi_anchor_expansion.py "$study/replay.json.gz" \
    --fixture "$study/fixture.json.gz" --pools "$study/pools.json.gz" \
    --output /tmp/kam-multi-anchor.json
diff -u "$study/summary.json" /tmp/kam-multi-anchor.json
```
