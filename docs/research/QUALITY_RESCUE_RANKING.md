# Can first-pass logits predict when Flash will rescue an answer?

On the existing intact QuALITY panel, a learned reranker of potential Flash rescues did not improve on a simple confidence threshold for the final-256 Flash window. It triggered 117 reruns, gaining eight correct answers and breaking eight previously correct answers. The margin rule triggered 23 reruns and gained eight correct answers. A label-aware ceiling shows there were more correctable decisions than either inference-visible policy found.

## Article-held-out replay

We fixed the predictors and decision rule before replaying 452 original-order questions from 113 articles with the same three Llama adapter seeds. Each of five folds held out whole articles by a deterministic hash. The other four folds fit the ranker and selected its cutoff. No fold used its own questions to fit either policy.

The fit predicts the signed accuracy change of a rerun with ridge regularization. Its input is the sparse top-two option gap, four-option softmax entropy, log prompt width and top option position. For a tie, position indicators divide evenly among tied options. We standardize on the fitting folds and use a fixed ridge penalty of 0.10 under the average squared loss. To decide which examples to rerun, rank positive predicted gains per millisecond of the full Flash pass. The margin policy ranks by ascending top-two gap.

Each rule selects a cutoff on its fitting articles under 99% of the corresponding Flash-only modeled prefill, maximizing accuracy and breaking ties by lower cost. The label-aware ceiling picks the cheapest true rescues under the same cost target per seed. Costs interpolate interleaved HPG batch-one p50 readings for each actual unpadded width and charge the full second pass. This models prefill cost and leaves dispatch overhead unmeasured. Accuracies use the original option order and count a top-score tie as incorrect. Paired intervals resample held-out articles with the fitted folds and three checkpoints fixed.

## Accuracy and prefill

| First pass | Rerun policy | Accuracy | Reruns | Modeled prefill |
| --- | --- | ---: | ---: | ---: |
| Full Flash | None | 60.91% | 0.00% | 305.16 ms |
| Final-256 Flash | None | 61.43% | 0.00% | 285.93 ms |
| Final-256 Flash | Margin | 62.02% | 1.70% | 291.26 ms |
| Final-256 Flash | Learned signed rescue | 61.43% | 8.63% | 300.61 ms |
| Final-256 Flash | Label-aware ceiling | 66.30% | 4.87% | 301.46 ms |
| Protected shortlist | None | 52.73% | 0.00% | 282.40 ms |
| Protected shortlist | Margin | 54.13% | 2.80% | 291.06 ms |
| Protected shortlist | Learned signed rescue | 55.46% | 12.46% | 302.15 ms |
| Protected shortlist | Label-aware ceiling | 62.09% | 9.37% | 301.62 ms |
| Output-selected budget | None | 51.99% | 0.00% | 280.37 ms |
| Output-selected budget | Margin | 52.73% | 1.92% | 286.06 ms |
| Output-selected budget | Learned signed rescue | 54.28% | 10.84% | 298.10 ms |
| Output-selected budget | Label-aware ceiling | 61.06% | 9.07% | 299.79 ms |

The output-selected policy had its own interleaved Flash reference at 303.14 ms. The full-Flash logits and accuracy were shared. Rerun rates can exceed the time share available for reruns because the ranking can favor shorter prompts.

The primary learned policy gave **−0.59 points [−1.62, +0.52]** relative to the margin rule with **9.35 ms** higher modeled prefill. Its paired difference versus full Flash was **+0.52 [−2.36, +3.24]**. The learned policy missed both criteria for selecting it: a positive lower interval limit over the margin rule and a lower limit of at least −2 points versus Flash. The output-selected learned policy gained 1.55 points [+0.15, +3.10] over its margin trigger, with 12.04 ms more modeled prefill. It still trailed Flash by 6.64 points [−10.69, −2.58]. The three first passes should be treated as related comparisons on the same source panel.

The final-window ceiling gained 66 correct predictions over its own first pass by deferring only true rescues. The learned policy triggered eight rescues and eight regressions. The remaining opportunity would require stronger signals that identify when a full-attention computation changes the answer constructively. A future test can evaluate first-pass internal diagnostics or a within-pass decision on independently sourced articles.

[Replay script](../../scripts/analyze_kam_rescue_ranking.py) · [Fold assignments, fit details and paired results](../../results/general_benchmarks_quality_native/rescue-ranking.json) · [Earlier single-split baseline](CONFIDENCE_FLASH_DEFERRAL.md)
