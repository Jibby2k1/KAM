# Allocate a fixed block budget across Llama layers

Selecting layers by how much extra blocks improve projected Flash outputs gave the strongest 8K retrieval. It reached 59.64% on HotpotQA versus 42.71% for Flash, and 55.21% on MuSiQue versus 55.47% for Flash. This gain was specific to the measured long retrieval tasks. PG-19 perplexity rose, and 4K MuSiQue declined.

Compare two development-selected schedules against a periodic schedule with the same number of layers at each budget. Keep the checkpoint, proxy scoring, block size, block-zero protection and softmax implementation fixed. A layer receives four or twelve blocks. Sixteen layers receive each allocation.

## Calibration and evaluation

- For the output-selected schedule, run Flash on 13 separate PG-19 development books. Reconstruct each projected attention output from the actual protected shortlist at both budgets on every 256th query starting at token index 511. Score the per-book difference in squared output error normalized by the Flash output energy at that layer. Give twelve blocks to the sixteen highest-gain layers, breaking ties in layer order.
- For the evidence-selected schedule, audit both prompt orders of 48 separate MuSiQue development questions at the final choice query. Rank the layers by the gain in dense probability assigned to earlier evidence outside block zero when routed at twelve blocks instead of four. Give twelve blocks to the sixteen highest-gain layers. Keep all calibration examples and selection hashes separate from test inputs.
- Compare each seeded frozen model on the independent 23-book PG-19, 128-question MuSiQue and 128-question HotpotQA panels. Report perplexity and mirrored accuracy separately at 4K and 8K. Compare against alternating four/twelve, uniform eight, and Flash. Report pairing by evaluation unit and matched full-model RTX PRO 6000 prefill timings.

The alternating and selected schedules have identical block counts at each query position over the full layer stack. Uniform eight has the same nominal average per-layer budget. Its total count is slightly higher at short prefixes where fewer than twelve blocks are visible. Independent GPU smoke tests reconstruct each returned shortlist with a Torch mask.

## Results on HPG

| Context | Policy | PG-19 perplexity | MuSiQue mirrored | HotpotQA mirrored | Prefill p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | Flash | 11.0198 | 74.48% | 62.24% | 194.94 ms |
| 4K | Uniform eight | 12.2512 | 59.64% | 43.49% | 186.15 ms |
| 4K | Alternating four/twelve | 12.4009 | 49.48% | 41.41% | 186.31 ms |
| 4K | Evidence-selected | 12.6321 | 51.04% | 44.27% | 186.44 ms |
| 4K | Output-selected | 12.4821 | 48.70% | 51.04% | 186.25 ms |
| 8K | Flash | 10.8083 | 55.47% | 42.71% | 448.04 ms |
| 8K | Uniform eight | 12.3469 | 50.78% | 49.22% | 406.79 ms |
| 8K | Alternating four/twelve | 12.5104 | 45.83% | 51.56% | 406.91 ms |
| 8K | Evidence-selected | 12.7301 | 48.96% | 57.03% | 406.67 ms |
| 8K | Output-selected | 12.5966 | 55.21% | 59.64% | 406.77 ms |

On 8K HotpotQA, output selection improved mirrored accuracy by 16.93 points over Flash [6.51, 27.34] and 10.42 over uniform eight [2.08, 18.75]. The difference from alternating was +8.07 points [−0.52, +16.41]. On 8K MuSiQue, it gained 9.38 points over alternating [0.79, 17.97] and differed from Flash by −0.26 [−11.55, +10.94]. On 4K MuSiQue, it lost 10.94 points to uniform eight [−20.05, −2.07]. At 4K, output selection improved HotpotQA by 9.64 points over alternating [1.04, 18.23] and remained 11.20 points behind Flash [−21.09, −1.30]. Paired intervals resample question families conditional on the three checkpoint seeds.

At 8K the output-selected policy took 9.21% less prefill time than Flash and increased perplexity by 16.55%. Relative to alternating, its mean token NLL difference was +0.0069 [−0.0006, +0.0137]. The evidence-selected policy raised NLL at both widths and improved no mirrored accuracy comparison against alternating with a paired interval above zero. Incremental memory was measured on the same batch-one GPU profile. Calibration relied on dense activations at 4K, and quality reused frozen independent prompt panels at two widths.

The output-selected budget justifies a locked test on intact QuALITY articles. Its opposite results at 4K and 8K make the width and task explicit parts of the reported tradeoff.

[Paired results and timing](../../results/general_benchmarks_layer_budget/summary.json) · [Locked allocations](../../results/general_benchmarks_layer_budget/layer-budget-selection.json) · [HPG receipts](../../results/general_benchmarks_layer_budget/scheduler.tsv)
