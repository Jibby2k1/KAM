# Can one-pass confidence pay for a second Flash prefill?

On held-out QuALITY articles, confidence-based reruns recovered little accuracy from the first sparse pass. The cost model leaves room to rerun only roughly 6–7% of the total Flash time. The best result came from using the final Flash window as the first pass, with very few questions sent to a second pass.

Use the frozen QuALITY predictions to test a deployment policy that runs a sparse prefill first and reruns full Flash only when the first answer has a low top-two logit margin. The basic budget charges both passes for every deferred request. This complements the within-pass Flash window, which changes the attention implementation before making a prediction.

## Offline protocol

- Use the published original option order for the reader decision and the margin. Thresholds may use two returned logits and the task's prefill budget. They use no answer labels at inference.
- Partition **articles** deterministically by hashed article ID into a calibration half and an evaluation half. Choose one threshold per first pass on calibration articles, pooling the three model seeds. Candidate thresholds are observed sparse top-two margins, plus zero deferrals. Choose highest standard accuracy subject to total predicted prefill no higher than Flash alone. Break ties by lower total prefill.
- Predict each policy's prefill for the exact prompt width by interpolating its eight interleaved batch-one measurements. The second pass reruns the full prompt. Use the matched Flash timing from the corresponding benchmark for each first pass.
- Report paired evaluation-article intervals, coverage, prefill and accuracy. Also compute a label-aware oracle that spends the same budget only on cases Flash rescues, choosing the cheapest rescues first. This gives a ceiling on standard accuracy at the measured cost model. Report the cheapest total prefill that would reach Flash accuracy with oracle decisions.

The comparison uses the same already-scored article benchmark. Threshold selection leaves evaluation articles out. The measured inputs are individual prefill timings and logits. Composite latency estimates use the stated interpolation and full-rerun model.

## Held-out evaluation

The thresholds used 228 questions from 57 calibration articles. Evaluation used 224 questions from the remaining 56 articles. Flash original-order accuracy on the evaluation half was 58.78% across the three seeds.

| First pass | First-pass accuracy | Rerun rate | Accuracy after reruns | Modeled prefill per question | Matched Flash prefill | Label-aware ceiling at Flash cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protected shortlist | 51.49% | 3.13% | 52.68% | 288.87 ms | 301.19 ms | 62.05% |
| Final Flash window | 59.67% | 1.34% | 60.12% | 286.31 ms | 301.19 ms | 64.29% |
| Output-selected budget | 51.79% | 8.48% | 52.98% | 305.59 ms | 299.24 ms | 61.90% |

The chosen threshold was zero logits for both the protected shortlist and the final window, so only top-score ties triggered a rerun. The output-selected threshold was 0.125 logits. It budgeted within the Flash prediction on calibration and exceeded it by 6.35 ms per question in evaluation.

The final-window rule was 1.34 points above Flash [−2.83, +5.21], with 14.88 ms less modeled prefill [−18.64, −10.89]. It recovered 0.45 points over its own first pass. The protected rule remained 6.10 points below Flash [−12.50, +0.15], and the output-selected rule was 5.80 below [−11.16, −0.45]. Intervals resample held-out articles conditional on the three seeds.

With labels available to the oracle, the minimum modeled cost to match Flash accuracy from protected shortlisting was 290.35 ms versus 301.19 ms for Flash. The gap between the learned threshold and oracle suggests value in signals that can predict a rescue while avoiding an unnecessary rerun. A width-dependent Flash window changes the first pass itself and avoids paying for a second full prompt on those requests.

[Frozen policy analysis](../../results/general_benchmarks_quality_native/deferral-policy.json) · [Underlying QuALITY evaluation](NATIVE_LONG_DOCUMENT_QA.md)
