# Can pretrained adapters use earlier MuSiQue evidence?

This diagnostic reuses the fixed-KL SmolLM2 and Llama checkpoints. Each question comes with two plausible MuSiQue endpoint passages. The correct and crossed earlier passages have exactly matched token counts within each backbone. In the crossed condition, the earlier passage comes from the paired question and points to the foil. Each question is presented with both endpoint orders to measure whether the choice follows the passage when its A/B letter changes. [Locked protocol](RESEARCH_PLAN.md#frozen-adapter-choice-on-matched-musique-endpoints) · [Earlier open-corpus experiment](../../reports/nonlinear_retrieval_natural/MUSIQUE_ENDPOINT_ROUTING.md)

The panel has 168 questions in 84 pairs across 35 relation families. The exact normalized audit found no matches to the three checked adapter-training inputs. The annotated bridging name appears in the correct earlier and endpoint passages for all questions, and the paired earlier passage contains the foil's name. This setup supplies the evidence and uses a fixed pair of endpoint candidates. [Panel manifest](../../results/general_benchmarks_pretrained_endpoints/fixture-manifest.json) · [Source overlap](../../results/general_benchmarks_pretrained_endpoints/source-overlap.json) · [Anchor audit](../../results/general_benchmarks_pretrained_endpoints/anchor-audit.json)

## Registered result

The model chooses A or B from the next-token logits. A correct question requires two correct choices after mirroring. Means use the three matched seeds. Intervals resample relation families conditional on those seeds.

| Backbone and earlier material | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, endpoints only | 2.98% | 2.78% | 2.58% | +0.20 [0.00, +0.70] pp |
| SmolLM2, verified earlier passage | 5.36% | 4.96% | 3.97% | +0.99 [−0.25, +3.06] pp |
| SmolLM2, crossed earlier passage | 1.19% | 0.60% | 0.60% | 0.00 [−0.61, +0.62] pp |
| Llama, endpoints only | 54.76% | 57.14% | 66.07% | −8.93 [−12.04, −5.63] pp |
| Llama, verified earlier passage | 55.95% | 73.41% | 83.53% | −10.12 [−16.31, −3.65] pp |
| Llama, crossed earlier passage | 24.40% | 29.76% | 29.76% | 0.00 [−2.42, +2.93] pp |

Verified earlier evidence raised the Llama KAM score by 16.27 points and dot by 17.46. The corresponding change in the KAM–dot gap was −1.19 points [−7.36, +6.14]. The larger dot score was already visible without the earlier passage. For SmolLM2, the two mirrored prompts elicited the same answer letter on about 95–96% of questions in the verified-earlier condition. This yields very low two-order accuracy even when the score margins vary with the content. [SmolLM2 summary](../../results/general_benchmarks_pretrained_endpoints/smollm2-summary.json) · [Llama summary](../../results/general_benchmarks_pretrained_endpoints/llama3-summary.json)

## Interpreting the two orders together

After inspecting the registered results, subtract the reversed-order `B − A` logit margin from the original-order margin. This cancels any additive letter preference common to the two orders. A positive difference selects the original B passage. A zero difference is unresolved and counted as incorrect. The combined rule uses two forward passes per decision.

| Backbone and earlier material | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, endpoints only | 57.74% | 53.37% | 54.56% | −1.19 [−4.92, +1.97] pp |
| SmolLM2, verified earlier passage | 66.07% | 67.86% | 67.26% | +0.60 [−2.32, +3.97] pp |
| Llama, endpoints only | 86.31% | 85.91% | 86.90% | −0.99 [−2.75, +1.28] pp |
| Llama, verified earlier passage | 95.83% | 95.83% | 95.44% | +0.40 [0.00, +1.53] pp |

The large raw Llama gap shrank under this two-pass rule. With true earlier evidence and an otherwise matched crossed prompt, the symmetrized choice was correct on the true prompt and selected the foil on the crossed prompt for 53.57% of KAM questions and 53.97% of dot questions. Both arms respond to the supplied evidence. The exclusive bridging term makes this a comparatively easy link-following test. The next locked control gives both earlier passages to the reader and mirrors their order independently of the endpoint order. [Two-earlier-passage control](RESEARCH_PLAN.md#locked-llama-control-with-two-earlier-passages)

## Both earlier passages visible

The same seven Llama checkpoints saw both early passages, with the verified one first or second and the endpoints independently reversed. Both candidate names now appear in the early evidence. The text and passage lengths were held fixed across the two early orders. The registered primary outcome requires four correct choices for a question, one for each presentation.

| Early evidence order and choice rule | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Verified early passage first, both endpoint choices correct | 10.12% | 17.86% | 29.17% | −11.31 [−17.49, −7.50] pp |
| Verified early passage second, both endpoint choices correct | 46.43% | 48.02% | 39.68% | +8.33 [+4.31, +11.98] pp |
| All four choices correct | 8.93% | 12.90% | 17.86% | −4.96 [−10.10, −1.58] pp |
| First, combined endpoint margins | 85.12% | 86.31% | 86.71% | −0.40 [−2.40, +2.63] pp |
| Second, combined endpoint margins | 74.40% | 74.01% | 76.79% | −2.78 [−7.26, +0.54] pp |
| Combined endpoint margins correct in both early orders | 70.24% | 71.63% | 73.61% | −1.98 [−6.63, +1.67] pp |

Moving the verified early passage from first to second shifted the raw KAM–dot gap by +19.64 points [14.74, 27.24]. The symmetric comparison shows that the model still responds to passage order after a common A/B margin is canceled. Comparisons on this panel depend on how choices are produced and how evidence is ordered. Run controlled order changes on independent relations before making a one-pass claim about open search. [Frozen manifest](../../results/general_benchmarks_pretrained_endpoints/both-earlier-manifest.json) · [Paired summary](../../results/general_benchmarks_pretrained_endpoints/llama3-both-earlier-summary.json) · [Numeric choices](../../results/general_benchmarks_pretrained_endpoints/llama3-both-earlier-outcomes.json.gz) · [HPG scheduler receipt](../../results/general_benchmarks_pretrained_endpoints/llama3-both-earlier-scheduler.tsv)

## Reproduce

The [evaluator](../../scripts/eval_kam_pretrained_endpoint_selection.py) prepares the pinned pairs, maintains matched token widths, and runs the frozen adapters using the [HPG driver](../../slurm/kam_pretrained_endpoint_selection.sbatch). The [summarizer](../../scripts/summarize_kam_pretrained_endpoint_selection.py) checks the checkpoint lineage, preserves mirrored presentation records, and replays the relation-family contrasts. [SmolLM2 numeric choices](../../results/general_benchmarks_pretrained_endpoints/smollm2-outcomes.json.gz) · [Llama numeric choices](../../results/general_benchmarks_pretrained_endpoints/llama3-outcomes.json.gz) · [SmolLM2 scheduler receipt](../../results/general_benchmarks_pretrained_endpoints/smollm2-scheduler.tsv) · [Llama scheduler receipt](../../results/general_benchmarks_pretrained_endpoints/llama3-scheduler.tsv)
