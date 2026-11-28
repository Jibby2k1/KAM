# Does the final Flash window recover use of distant evidence?

On the locked 4K MuSiQue holdout, the final-256 policy improved the original endpoint choice. This experiment tests whether that improvement depends on the annotated earlier passage. [Query-window comparison](QUERY_WINDOW_FLASH.md)

## Fixed comparison

- Keep both endpoint passages, the question, the first 64 tokens and the full 4,096-token layout unchanged. Replace the part of the earlier passage outside the first block with a same-length native distractor segment from the prompt. Verify the annotated bridge mention is fully outside the first block and absent after the replacement.
- Use the exact same replacement tokens in an equal-length neighboring background span for the control. Reuse the same donor in both endpoint orders. Check that these changes touch only the registered token intervals.
- Compare Flash, the protected eight-block shortlist and the final-256 Flash window over all 32 layers, with the same three frozen adapter seeds. Keep the original unedited outcomes from the matched archives.
- Score the mirrored logit margin for the original gold endpoint, then compare the change from replacing earlier evidence against changing the background. The edited prompts test dependence on supplied evidence. The original endpoint is a reference target for the perturbation rather than a fresh annotation of the edited prompt.

The primary contrast is the tail-window's extra drop relative to the protected shortlist when earlier evidence is replaced. Pair by question and resample relation families. Treat the original and background-control scores as checks on the direction and specificity of the effect.

## Matched result

The verified panel contains 269 questions in 225 relation families with the bridge mention fully outside the first block and absent after replacement.

| Policy | Original mirrored gold preference | Background replaced | Earlier evidence replaced | Earlier minus background, signed margin [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Flash | 67.29% | 67.41% | 56.13% | −0.442 [−0.534, −0.355] |
| Protected shortlist | 55.51% | 53.28% | 54.15% | −0.016 [−0.059, +0.027] |
| Final 256 | 60.97% | 59.36% | 56.75% | −0.127 [−0.167, −0.089] |

In the original-minus-edited control check, the background-only margin changes were small and each paired interval spanned zero. The final-window policy's drop from editing early evidence exceeded the protected shortlist's by 0.111 logits [+0.054, +0.167]. Its drop remained 0.315 logits smaller than Flash [−0.386, −0.248]. The direction of both margin effects held in all three seeds.

This intervention supports partial recovery of reliance on distant evidence with the final Flash window. The original earlier passage still affected full Flash much more strongly. A first-block-preserving prefix test now checks whether a fixed token budget can restore that access across more query positions. The source annotations are used only offline to measure these effects.

Run [the evaluator](../../scripts/kam_distant_evidence_ablation.py) to prepare from the frozen evidence spans and score each of the nine seed/attention combinations. Use [the summarizer](../../scripts/summarize_kam_distant_ablation.py) with the three original outcome archives. [Perturbation manifest](../../results/general_benchmarks_distant_ablation/ablation-manifest.json) · [Summary](../../results/general_benchmarks_distant_ablation/summary.json) · [Replay archive](../../results/general_benchmarks_distant_ablation/outcomes.json.gz) · [HPG jobs](../../results/general_benchmarks_distant_ablation/scheduler.tsv)
