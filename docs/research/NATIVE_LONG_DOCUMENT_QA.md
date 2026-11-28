# Long-document comprehension on intact QuALITY articles

On 452 questions from 113 intact articles, the final Flash window reached 66.59% rotation-balanced accuracy versus 69.10% for Flash and 58.04% for protected shortlisting. Its 8,189-token median prefill was 8.26% faster than Flash. The same three frozen Llama checkpoints were used throughout.

Test full Flash, protected shortlisting and the final-window fallback on native four-option comprehension questions. Use complete articles in their published order, with no synthetic distractors or moved evidence. QuALITY's hard questions were usually missed under short reading limits by annotators who could answer them given enough time. [QuALITY dataset and evaluation](https://aclanthology.org/2022.naacl-main.391/)

## Frozen panel

- Use the official QuALITY v1.0.1 HTML-stripped development file, whose answers are published. Keep every article with an eligible prompt at or below 8,192 tokens. Select up to four questions per article by hashed question ID, up to two hard and two other when available. Record any width and option exclusions before model inference. [Publisher release](https://github.com/nyu-mll/quality/tree/main/data/v1.0.1)
- Carry the same evidence and four options through four cyclic option orders. Report standard original-order multiple-choice accuracy, accuracy after averaging the four sets of logits in their underlying answer order, and accuracy requiring all four orders. Use full Flash on the same questions with the article removed to measure the benefit of supplied context.
- Compare the same three frozen Llama checkpoints. Keep the original question wording, article and answer options. The final dense window starts at the last tile boundary at or before 256 tokens from the end, so native unpadded prompts use 256–287 Flash queries. Check the partial last tile against a full Flash reference before evaluation.
- Resample articles for paired 95% intervals, conditional on the three seeds. Report all questions, the publisher hard subset, and prompt widths up to and above 4,096. Profile eight matched native lengths with 15 interleaved trials per policy on the same HPG RTX PRO 6000.
- Run the locked output-reconstruction allocation from the layer-budget study on this same panel. The allocation was fixed before reading the native article scores. Give the transfer its own interleaved Flash and protected timing controls at the same eight lengths. [Locked development schedule](LAYER_BLOCK_BUDGET.md)

## Matched native results

The panel has 226 publisher-hard questions and a median width of 6,604.5 tokens. It uses 414 unique unpadded lengths. Every cyclic order of a question had the same length. Fifty-seven source questions exceeded the width eligibility bound. The released development split supplies the gold answers.

| Attention | Original order | Rotation-balanced | Correct in all four orders | Hard rotation-balanced |
| --- | ---: | ---: | ---: | ---: |
| Flash with question only | 37.61% | 41.96% | 15.27% | 38.35% |
| Flash with full article | 60.91% | 69.10% | 40.41% | 63.72% |
| Protected shortlist | 52.73% | 58.04% | 31.64% | 52.21% |
| Final Flash window | 61.43% | 66.59% | 39.16% | 60.18% |
| Output-selected layer budget | 51.99% | 58.78% | 30.24% | 51.92% |

The balanced final-window difference was −2.51 points versus Flash [−5.46, +0.22] and +8.55 versus protected shortlisting [4.57, 12.54]. On the 332 prompts longer than 4,096 tokens, the corresponding differences were −2.71 [−6.43, +0.90] and +6.33 [1.51, 11.35]. Flash gained 27.14 points from the article over the question-only control [21.98, 32.37]. These paired intervals resample articles conditional on the three checkpoints.

The four-choice log loss after rotation balancing was 0.820 with Flash, 0.863 with the final window and 1.046 with protected shortlisting. The paired final-window increase over Flash was 0.044 [0.006, 0.084]. The log loss normalizes over the four answer choices, with all option positions represented equally.

At the three longest profiled widths, the final window took 351.57 ms at 7,047 tokens, 384.87 ms at 7,428, and 415.91 ms at 8,189. Flash took 379.15, 415.92 and 453.37 ms respectively. Timing is per presentation. Original-order accuracy uses one presentation and rotation-balanced accuracy uses four for every policy.

The fixed reconstruction-selected layer budget lost 10.32 rotation-balanced points to Flash [−14.75, −5.90] and gained 0.74 versus uniform protected shortlisting [−1.47, +2.95]. Its separate 8,189-token profile took 407.29 ms versus 449.75 ms for Flash, with 935.14 MiB of incremental peak allocation for all three policies. The earlier 8K HotpotQA effect remains specific to that retrieval panel. [Fixed schedule and prior 8K comparison](LAYER_BLOCK_BUDGET.md)

### Exploratory human context ratings

For 120 questions, a majority of untimed validators recommended using at least a third of the article. On this set, rotation-balanced accuracy was 67.78% with Flash, 66.39% with the final window, and 62.50% with protected shortlisting. The paired final-window difference versus protected was +3.89 points [−2.26, +10.36], compared with +10.24 [5.64, 14.86] on the remaining 332 questions. The ratings describe the amount of text validators reported needing. [Publisher annotation definitions](https://github.com/nyu-mll/quality#data-readme)

[Frozen scores, calibration and profiling](../../results/general_benchmarks_quality_native/summary.json) · [HPG receipts](../../results/general_benchmarks_quality_native/scheduler.tsv)
