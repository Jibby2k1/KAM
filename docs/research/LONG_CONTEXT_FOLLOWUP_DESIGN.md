# Design the next matched long-context tests

The first 8K HotpotQA panel gave a large advantage to the output-selected layer budget. The same schedule had lower accuracy on intact QuALITY. Keep the task, width, inference rule and comparison fixed before collecting more model predictions.

## Experiment order

1. Score the six completed sparse continuation checkpoints under both Flash and protected shortlisting. Keep the fixed 4K/8K PG-19, MuSiQue and HotpotQA panels. Use the train-by-evaluation interaction as the primary diagnostic of sparse-specific recovery. [Matched training receipts](SPARSE_ADAPTER_RECOVERY.md)
2. Run the prepared 256-question HotpotQA replication. The first comparison is output-selected minus Flash at 8K. The equal-budget alternating comparison tests whether the selection criterion helps. The larger design below gives separate targets for detecting a smaller effect.
3. Test the locked four/eight/twelve allocation from development data on a fresh live reader holdout. Include PG-19-like language loss and native question accuracy together. Use uniform eight as the same-average-budget baseline. Retain Flash and the existing final window as references. [Locked frozen-activation analysis](ATTENTION_MASS_AND_RECONSTRUCTION.md)
4. Profile the variable-length kernel once matched output parity holds. If the exact mask and all scored outputs agree, report cold compilation, warm prefill, throughput and peak memory. Test a retained-mass certificate on sampled actual query/key states before using it as a live fallback. [Certificate derivation and controlled check](SHORTLIST_MASS_CERTIFICATE.md)

## Sample sizes from the observed pair variation

For a case $i$, average its matched score difference over the three completed checkpoint seeds, giving $d_i$. Group cases by article, book or question family. For cluster $g$, use residual score $T_g-N_g\bar d$, where $T_g=\sum_{i\in g}d_i$ and $N_g$ is its case count. With $G$ clusters and $N$ cases, estimate paired variance by

$$
\widehat{\operatorname{Var}}(\bar d)=
\frac{G}{G-1}\frac{\sum_g(T_g-N_g\bar d)^2}{N^2}.
$$

Scale variance by the number of future independent clusters using the observed $G\widehat{\operatorname{Var}}$ as a planning estimate. The table uses 80% power and a 5% significance level for one prespecified comparison. For a superiority test, the gap is the true difference to detect. For noninferiority, the gap is a tolerance and the reference scenario assumes equal underlying performance.

| Comparison | Illustrative target | Independent clusters | Approximate examples | Profile-equivalent GPU hours |
| --- | --- | ---: | ---: | ---: |
| 8K HotpotQA, output-selected versus Flash | Detect 5 points | 1,149 question families | 1,149 questions | 1.64 |
| Same 8K HotpotQA comparison | Detect 8 points | 449 families | 449 questions | 0.64 |
| Same 8K HotpotQA comparison | Detect 10 points | 288 families | 288 questions | 0.41 |
| MuSiQue, change in the final-window effect from 4K to 8K | Detect 8 points | 519 families | 528 questions | 1.05 |
| Intact-article final-window noninferiority | Allow 2 points at equal true accuracy | 368 articles | 1,472 questions | 2.90 |
| Same intact-article comparison | Allow 3 points at equal true accuracy | 164 articles | 656 questions | 1.29 |
| 8K language loss noninferiority | Allow 0.01 mean NLL at equal true loss | 205 books | 205 books | — |

The prior 128-question HotpotQA run observed a 16.93-point difference. A 256-question replication has useful sensitivity to a large effect and less power for a five-point effect. Expand question-family coverage if the fresh effect is closer to five points.

On QuALITY, the observed final-window difference was −2.51 points. If that difference were the true effect, a two-point noninferiority condition would fail and a three-point margin would require about 6,063 independent articles under the same planning calculation. A five-point noninferiority tolerance under that same observed effect requires about 237 articles. Use independent article sources when increasing the number of questions from the released panel would mainly repeat the same article clusters.

On PG-19, final-window NLL exceeded Flash by 0.102 at 8K. The 0.01 allowance describes a future candidate expected to have equal loss. With the observed gap as the true value, the candidate remains outside that margin as the number of books grows.

The Q/A cost estimates sum the measured per-presentation prefills for two matched arms, all three seeds, and the required two or four answer orders. They describe GPU prefill time under the previous RTX PRO 6000 profiles. The larger HotpotQA comparison can separately add the alternating and uniform controls to study mechanism.

[Exact inputs and calculations](../../results/general_benchmarks_layer_budget/followup-design.json)
