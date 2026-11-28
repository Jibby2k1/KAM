# Cross-corpus transfer to long HotpotQA prompts

The fixed final-256 Flash window saved 3.02% of prefill time at 4K and 8.11% at 8K on fresh HotpotQA prompts. At 4K it recovered part of the protected shortlist's retrieval gap on unlinked questions and remained 12.76 percentage points below full Flash overall. At 8K the paired comparisons with both references were inconclusive.

## Fixed protocol

- Use hard bridge questions with two annotated support documents. Require the earlier passage to mention the gold endpoint title and the annotated endpoint sentence to contain the answer. Choose the foil from each question's native distractors.
- Split the 128-question panel evenly between cases where the earlier passage also mentions the foil title and cases where it does not. Hold the evidence, endpoints and answer orders fixed at 4K and 8K. Extend width using more of the same distractor stream. Require the earlier anchor and endpoint answer to remain visible after tokenization.
- Reserve question IDs detected in prior versioned KAM results before selecting candidates and require disjoint support titles within the panel. The unused publisher development rows contained only 41 linked candidates before tokenizer checks, so the locked panel draws from the first publisher training shard. Check support and foil overlap against the adapter training sources.
- Score original endpoint choice by the mirrored margin over both placements with the same three frozen Llama checkpoints. Report paired question intervals overall and by linked stratum. Interleave batch-one prefill timings on the same HPG RTX PRO 6000.

The corpus provides native Wikipedia distractors and annotated multi-hop evidence. The long prompts are constructed from these materials. This comparison measures conditional endpoint choice with both endpoints already supplied. [HotpotQA](https://aclanthology.org/D18-1259/)

## Results on HPG

| Context | Earlier passage also names foil | Flash | Protected shortlist | Final-256 Flash |
| --- | --- | ---: | ---: | ---: |
| 4K | All 128 | 62.24% | 43.49% | 49.48% |
| 4K | Linked (64) | 52.08% | 40.10% | 39.58% |
| 4K | Unlinked (64) | 72.40% | 46.88% | 59.38% |
| 8K | All 128 | 42.71% | 49.22% | 47.14% |
| 8K | Linked (64) | 31.77% | 44.79% | 43.75% |
| 8K | Unlinked (64) | 53.65% | 53.65% | 50.52% |

At 4K, the final window's paired difference was −12.76 points [−20.05, −5.47] against Flash and +5.99 [−1.82, +13.80] against the protected shortlist. Relative to protected shortlisting, the linked difference was −0.52 [−10.42, +8.85] and the unlinked difference was +12.50 [0.00, +25.00]. At 8K the paired overall differences were +4.43 points [−4.69, +13.54] against Flash and −2.08 [−10.94, +6.77] against protected shortlisting. Intervals resample question families and are conditional on the three checkpoint seeds. The source audit found one exact passage match with the Wikipedia adapter-training source. Excluding that question gave −12.07 points against Flash at 4K and +4.20 at 8K.

| Context | Flash p50 | Protected p50 | Final-256 p50 | Incremental allocation, all three |
| --- | ---: | ---: | ---: | ---: |
| 4K | 194.51 ms | 186.10 ms | 188.63 ms | 469.14 MiB |
| 8K | 447.06 ms | 405.79 ms | 410.80 ms | 935.17 MiB |

Quality used seeds 731201, 731211 and 731221. Timing used seed 731201 and 15 interleaved trials per width and policy. The paper claim remains a measured speed–quality tradeoff. Both native corpora show a loss relative to Flash at 4K, and this HotpotQA panel gives wider comparative uncertainty at 8K.

HPG jobs `53195801`, `53195802` and `53195803_0–8` completed. The archive contains the mirrored margins, raw timing trials, reservations, hashes and replayable metrics. [Experiment records](../../results/general_benchmarks_long_hotpot/summary.json) · [Scheduler receipt](../../results/general_benchmarks_long_hotpot/scheduler.tsv)
