# Which part of the adapter transfers?

The current Llama adapters change two things at once. Dot attention can reread prompt tokens and mixes all eligible values. KAM reads a fixed bank and mixes four values. Both score by dot products. Softmax and hard selection make their complete mappings nonlinear. The ability to express a feature interaction, the data required to learn it, and its performance on new examples are separate questions. [Matched dense retrieval](MUSIQUE_CACHED_RETRIEVAL.md) · [Adapter implementation](../../scripts/kam_pretrained_adapters.py)

## Frozen intervention on the read budget

Use the same three KAM and dot seeds and untouched Llama 3 8B. Supply the verified first passage and ask for the endpoint in both A/B orders. Replay the questions for which the prior dense retriever found the annotated first hop in both pool orders. Check native margins against those earlier results. Keep the publisher's training and development replay strata separate.

The independent question cohort takes one eligible question per new relation family from the publisher's development split. Exclude questions, normalized wordings, supporting titles and relation families from the earlier paired panels, plus question IDs from the title-field study. Select a native nonsupporting foil using BM25 over the question and verified anchor. Foils containing an accepted answer are ineligible. Prefer a foil mentioning the same anchor, and report that subset separately. Apply the same 512-token display and gold-visibility rules to every checkpoint. This cohort changes relation families and the foil construction together, so transfer to it is the primary generalization check and its absolute scores are descriptive across panel types. [MuSiQue source](https://github.com/StonyBrookNLP/musique) · [Frozen numeric manifest](../../results/general_benchmarks_routing_bottlenecks/manifest.json)

| Checkpoint | Intervention | Comparison |
| --- | --- | --- |
| Dot | Native SDPA, explicit dense, explicit top four, explicit top 16 | The explicit paths use the same projected live keys, values, causal mask and score arithmetic. Native SDPA measures the effect of changing the kernel alone. |
| KAM | Native top four, top 16, full 256-entry bank | Keep the bank, scores, temperature, projection and values fixed. Increase the number of entries contributing to the output. |
| Untouched Llama | Native | Check the base model's endpoint recovery. |

Measure endpoint accuracy in both orders, answer-slot disagreement, paired family intervals and warmed model-call latency. Treat the frozen interventions as sensitivity tests. Each checkpoint was trained with its native selection rule. A wider bank losing accuracy could reflect how the checkpoint learned to rely on sparse gating. Explicit attention scores all eligible keys, so its timings characterize this diagnostic implementation. A divergence between native and explicit dense should be reported before interpreting the top-four contrast. [Top-k attention reference](https://aclanthology.org/2021.sustainlp-1.5/) · [Evaluator](../../scripts/probe_kam_routing_bottlenecks.py)

### Read-budget results

| Reader and rule | Prior larger cohort, 78 questions | Fresh development cohort, 96 families | Fresh same-anchor foil, 24 families |
| --- | ---: | ---: | ---: |
| Untouched Llama | 41.03% | 55.21% | 45.83% |
| Dot, native SDPA | 72.22% | 73.61% | 54.17% |
| Dot, explicit dense | 72.22% | 74.31% | 54.17% |
| Dot, explicit top four | 74.36% | 71.88% | 54.17% |
| Dot, explicit top 16 | 74.36% | 75.00% | 56.94% |
| KAM, native top four | 58.97% | 68.40% | 47.22% |
| KAM, top 16 | 58.97% | 68.75% | 47.22% |
| KAM, full bank | 58.97% | 68.40% | 47.22% |

All native replay margins matched the previous dense-retrieval run exactly where the annotated first hop was recovered in both orders. The fresh source audit found zero exact normalized matches against the frozen adapter-training corpora. Native and explicit dense dot agreed on 569 of 576 fresh presentations including ties, with a maximum A/B margin change of 0.25. The explicit top-four versus explicit dense contrast was −2.43 points [−6.94, +2.08] on the fresh cohort. The small replay development set gave 93.33% explicit dense dot and 80.00% explicit top-four dot, with ten questions in four families. This leaves room for sparse selection to matter on some inputs. Widening the KAM bank yielded zero aggregate gain on the two larger cohorts. The result shifts attention toward what the current bank encodes and where it reads from. Those factors still require a matched training intervention. [Paired results](../../results/general_benchmarks_routing_bottlenecks/summary.json) · [Numeric outcomes](../../results/general_benchmarks_routing_bottlenecks/outcomes.json.gz) · [Overlap audit](../../results/general_benchmarks_routing_bottlenecks/source-audit.json) · [HPG receipts](../../results/general_benchmarks_routing_bottlenecks/scheduler.tsv)

## Matched training to isolate nonlinear scoring

### Fixed-embedding generalization probe

Before retraining Llama, use frozen E5-small-v2 representations to isolate the learned query map. Each pool contains one verified first hop and 16 native distractors. The two arms use the same 32-wide residual branch, two bias-free matrices, zero-initialized output, normalized queries, fixed passage keys and common score scale. The identity and SiLU activation are the only difference. Use three paired seeds and nested pools of 64, 256 and 512 distinct training questions, with six passes through each pool and the same minibatch order within each seed. Measure training NLL and held-out recall at one and three against the frozen E5 reference. Reserve new questions globally, hold out supporting article titles, and score the 96 held-out publisher development families separately. Also score new questions drawn from the larger fit pool's families and report their family overlap at each exposure. This measures learning and first-hop transfer in a fixed representation. It is a separate experiment from the Llama adapter intervention. [Fit and holdout manifest](../../results/general_benchmarks_routing_bottlenecks/activation-manifest.json) · [Query-map probe](../../scripts/probe_kam_query_activation.py)

| Distinct fit questions | Linear training NLL | SiLU training NLL | Linear unseen-family R@1 | SiLU unseen-family R@1 | SiLU minus linear |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.9045 | 1.1303 | 72.22% | 76.04% | +3.82 [−0.35, +8.33] pp |
| 256 | 0.8553 | 0.8681 | 71.88% | 70.14% | −1.74 [−3.82, −0.35] pp |
| 512 | 0.9402 | 0.9621 | 70.49% | 70.83% | +0.35 [−1.04, +2.08] pp |

Frozen E5 achieved 76.04% unseen-family R@1. At 512 questions, both maps reduced held-out NLL from 1.7371 to about 1.23 while lowering R@1. R@3 rose from 91.67% to 92.36% for linear and 92.71% for SiLU. Both trained maps gave 90.36% on the 128 held-out questions from the larger fit pool's families, versus 91.41% for frozen E5. Only 55 and 118 of those questions belong to a family represented at the 64- and 256-question budgets, respectively. These results test added nonlinearity on a fixed contextual encoder using one shared optimizer recipe. They leave open whether a different learning schedule or end-to-end representation supports better transfer. [All seeds and ranks](../../results/general_benchmarks_routing_bottlenecks/activation-results.json) · [HPG receipt](../../results/general_benchmarks_routing_bottlenecks/activation-scheduler.tsv)

### Live-token Llama follow-up

Once the frozen check characterizes acute read-budget sensitivity, compare a four-cell live-token design: soft and top-four selection crossed with an identity or SiLU query branch. Start from the same frozen backbone and share key/value projection architecture, positions, masks, width, normalization and score temperature. Give both query branches the same two bias-free matrices and zero-initialized residual output, differing only in activation. Keep the same training examples, order, optimizer, token budgets and seeds. Include dense dot and untouched Llama references, and compare against the original fixed bank as its own system.

Measure quality at logarithmically spaced training exposures, with one predeclared main checkpoint. Keep fitting and tuning data apart from held-out relation families, source articles and an independent corpus. For each holdout, change one factor at a time between paraphrase, distractor count and evidence distance. Record next-token loss, first-hop-conditioned endpoint recovery across orders, performance on input-preserving evidence swaps, route turnover and effective support counts. Where softmax temperature changes sharpness without changing rank, report ranking and calibration separately. If a nonlinear arm leads at equal training loss while consuming more data to reach that loss, report both the transfer and sample-efficiency effects. A gain on the shifted cohorts at matched exposure is the evidence for improved generalization. [Earlier linearized controls](../../reports/nonlinear_retrieval/FIXED_FLOP_DEVELOPMENT.md) · [Prior text data regimes](../../reports/nonlinear_retrieval/TEXT_DATA_REGIMES.md)
