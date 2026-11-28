# KAM benchmark scorecard

November 28, 2026 · [Experiment index](RESULTS.md) · [Dataset sources and splits](DATASET_PORTFOLIO.md)

The archive contains direct evaluations of small KAM checkpoints, retrieval pipelines with frozen pretrained encoders and readers, and deterministic controls. Each row identifies the comparison and evaluation slice. Scores are paired within rows and cover different systems and splits across rows.

## General model benchmarks

| Capability | Benchmark | Metric | KAM (T-KAM-F) | Dot (T-WIDE) | Full prompt, n |
| --- | --- | --- | ---: | ---: | ---: |
| Language modeling | Tiny Shakespeare | Validation NLL and character perplexity | Development score below | Development score below | — |
| Language modeling | [WikiText-103](https://arxiv.org/abs/1609.07843) | Held-out NLL and perplexity | Unscored | Unscored | — |
| Broad knowledge | [MMLU](https://arxiv.org/abs/2009.03300) | Multiple-choice accuracy | 24.20% (3,398/14,042) | 24.60% (3,454/14,042) | 1,233 |
| Graduate science | [GPQA Diamond](https://arxiv.org/abs/2311.12022) | Multiple-choice accuracy | 26.26% (52/198) | 23.74% (47/198) | 1 |
| Mathematical reasoning | [GSM8K](https://arxiv.org/abs/2110.14168) | Strict final-answer exact match | 0.00% (0/1,319) | 0.00% (0/1,319) | 3 |
| Code generation | [HumanEval](https://arxiv.org/abs/2107.03374) | Unit-test pass@1 | 0.00% (0/164) | 0.00% (0/164) | 1 |
| Code generation | [MBPP](https://arxiv.org/abs/2108.07732) | Unit-test pass@1 | 0.00% (0/500) | 0.00% (0/500) | 432 |
| Repository software engineering | [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) | Issues resolved with a specified agent harness | Unscored | Unscored | — |

The five general benchmarks use the first preregistered pair of TinyStories V2 checkpoints, with about 10M parameters and 50M training bytes each. Both models used the same zero-shot prompts on HPG L4 GPUs. Each forward keeps the most recent 128 UTF-8 bytes. The last column counts initial prompts that fit in full. Read these as scores for the specified checkpoint pair and prompt protocol. [Evaluation protocol and results](GENERAL_BENCHMARKS_PROTOCOL.md) · [Numerical summary](../../results/general_benchmarks/summary.json)

QASPER measures answers grounded in scientific papers. BRIGHT includes earth science document retrieval. Both are scored below.

### Longer context and broader data

These are newly trained, approximately 10M-parameter models. Each saw the same 134.2M bytes at a 2,048-byte context. The training mixture included prose, Python, and arithmetic. Earlier 128-byte scores use TinyStories checkpoints.

| Benchmark | Metric | KAM, 128 B | Dot, 128 B | KAM, 2,048 B | Dot, 2,048 B | Full 2,048-byte prompts |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | Pass@1 | 0.00% (0/164) | 0.00% (0/164) | 0.00% (0/164) | 0.00% (0/164) | 164/164 |
| MBPP | Pass@1 | 0.00% (0/500) | 0.00% (0/500) | 0.00% (0/500) | 0.00% (0/500) | 500/500 |
| GPQA Diamond | Accuracy | 26.26% (52/198) | 23.74% (47/198) | 23.74% (47/198) | 26.26% (52/198) | 197/198 |
| MMLU | Accuracy | 24.20% (3,398/14,042) | 24.60% (3,454/14,042) | 23.09% (3,242/14,042) | 22.92% (3,218/14,042) | 13,840/14,042 |
| GSM8K | Strict answer EM | 0.00% (0/1,319) | 0.00% (0/1,319) | 0.23% (3/1,319) | 0.68% (9/1,319) | 1,319/1,319 |

[Longer-context protocol, diagnostics, and training metrics](GENERAL_BENCHMARKS_LONG.md) · [Machine-readable results](../../results/general_benchmarks_long/summary.json)

### Learning-rate × KL factorial

The three seeds and five benchmarks were held fixed across all conditions within each backbone. Adapter cells list KAM / dot.

**SmolLM2-1.7B-Instruct**

| Benchmark | Untouched | High LR KAM / dot | High LR + KL KAM / dot | Low LR KAM / dot | Low LR + KL KAM / dot |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 28.66% | 30.89% / 29.07% | 28.25% / 29.27% | 28.05% / 27.03% | 27.64% / 28.46% |
| MBPP | 38.00% | 36.87% / 32.33% | 38.93% / 37.87% | 37.20% / 33.67% | 38.13% / 37.07% |
| GPQA Diamond | 25.76% | 24.92% / 24.41% | 26.60% / 26.26% | 26.77% / 23.74% | 26.26% / 23.06% |
| MMLU | 42.50% | 38.45% / 39.16% | 42.43% / 42.90% | 42.50% / 36.51% | 42.54% / 42.09% |
| GSM8K | 31.77% | 12.31% / 6.92% | 30.33% / 28.35% | 30.00% / 3.59% | 30.50% / 27.29% |

**Llama 3 8B Instruct**

| Benchmark | Untouched | High LR KAM / dot | High LR + KL KAM / dot | Low LR KAM / dot | Low LR + KL KAM / dot |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 47.56% | 44.11% / 47.97% | 51.02% / 49.80% | 49.59% / 48.58% | 49.59% / 50.00% |
| MBPP | 49.20% | 41.67% / 46.13% | 48.73% / 49.27% | 49.20% / 46.93% | 48.80% / 48.27% |
| GPQA Diamond | 29.80% | 28.28% / 29.63% | 28.62% / 29.46% | 28.28% / 27.61% | 28.79% / 28.45% |
| MMLU | 61.19% | 60.42% / 60.03% | 61.24% / 61.09% | 61.25% / 59.83% | 61.31% / 61.15% |
| GSM8K | 62.55% | 48.98% / 57.47% | 64.42% / 63.08% | 74.35% / 52.82% | 62.75% / 65.30% |

[Factorial report, interactions, and reproducibility records](GENERAL_BENCHMARKS_FACTORIAL.md)

### Intermediate KL at the lower learning rate

The same three seeds were evaluated at learning rate 0.00008 and KL weight 0.5. Cells list KAM / dot.

| Benchmark | SmolLM2-1.7B-Instruct | Llama 3 8B Instruct |
| --- | ---: | ---: |
| HumanEval | 28.25% / 25.41% | 52.03% / 50.00% |
| MBPP | 38.13% / 35.20% | 48.33% / 48.93% |
| GPQA Diamond | 26.43% / 23.40% | 27.95% / 30.81% |
| MMLU | 42.41% / 39.42% | 61.40% / 60.57% |
| GSM8K | 30.02% / 9.58% | 73.36% / 71.49% |

[Paired comparisons to KL weights 0 and 4](GENERAL_BENCHMARKS_DIAGNOSTICS.md#intermediate-kl-at-the-lower-learning-rate) · [GSM8K answer-format sensitivity](GENERAL_BENCHMARKS_DIAGNOSTICS.md#switching-the-requested-gsm8k-marker)

### Independent arithmetic transfer

The frozen low-rate Llama checkpoints and their untouched reference were evaluated on all 1,000 SVAMP problems. Scores use strict marked numeric exact match. Adapter cells average the same three seeds.

| Untouched | KL weight 0, KAM / dot | KL weight 0.5, KAM / dot |
| ---: | ---: | ---: |
| 80.90% | 80.57% / 62.93% | 80.00% / 77.43% |

[Transfer protocol, operator breakdown, and quantity contrasts](GENERAL_BENCHMARKS_DIAGNOSTICS.md#arithmetic-transfer-to-svamp)

### Development-selected KL on MATH-500

The KL weight was selected separately for each backbone on 512 held-out GSM8K development problems and shared between KAM and dot. The fresh test uses the 311 MATH-500 problems with integer labels. Scores are exact match averaged across the same three seeds. Both requires the same item to be correct under both answer formats.

| Backbone and selected KL | Format | Untouched | KAM | Dot |
| --- | --- | ---: | ---: | ---: |
| SmolLM2, KL 4 | `####` | 12.22% | 13.72% | 9.43% |
| SmolLM2, KL 4 | `Answer:` | 8.68% | 9.43% | 9.32% |
| SmolLM2, KL 4 | Both | 4.18% | 4.82% | 4.50% |
| Llama 3, KL 0.5 | `####` | 24.44% | 23.26% | 27.33% |
| Llama 3, KL 0.5 | `Answer:` | 29.26% | 26.26% | 26.26% |
| Llama 3, KL 0.5 | Both | 19.29% | 18.11% | 21.97% |

[Development selection, paired intervals, and test receipts](MATH_KL_TRANSFER.md)

### Full MATH-500 with a neutral answer prompt

All 500 problems use one prompt, no required answer marker, a 1,024-token cap, and the same Math-Verify grader. The KL weights were locked on development before this evaluation. Adapter scores are means across the same three seeds.

| Backbone and selected KL | Untouched | KAM | Dot | KAM minus dot (95% paired item interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, KL 4 | 22.80% | 23.53% | 22.80% | +0.73 [−1.40, +2.87] pp |
| Llama 3, KL 0.5 | 29.40% | 27.67% | 28.13% | −0.47 [−3.20, +2.27] pp |

[Full MATH-500 report and numeric outcomes](FULL_MATH500.md) · [Research plan](RESEARCH_PLAN.md)

### GSM8K prompt and grader intervention

On 512 unused GSM8K questions, both Llama adapters answered the same marked prompts at KL 0.5. Changing only the grader on the resulting text reversed the ranking.

| Prompt | Shared equivalence KAM / dot | Strict marker KAM / dot |
| --- | ---: | ---: |
| Neutral | 80.27% / 79.23% | — |
| `####` | 87.37% / 84.64% | 53.45% / 58.92% |
| `Answer:` | 86.59% / 84.11% | 46.48% / 51.89% |

[Both backbones, paired effects, and per-item records](GSM8K_FORMAT_EFFECTS.md)

### Balanced MATH answer-type replication

The neutral protocol was repeated on 478 disjoint items from the original MATH test collection, in 239 integer/noninteger pairs matched on subject and difficulty.

| Backbone and selected KL | Untouched | KAM | Dot | KAM minus dot (95% paired item interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, KL 4 | 26.57% | 25.73% | 25.45% | +0.28 [−2.23, +2.72] pp |
| Llama 3, KL 0.5 | 32.43% | 29.08% | 28.17% | +0.91 [−1.88, +3.63] pp |

The registered Llama difference in the KAM–dot gap between noninteger and integer questions was −1.53 points [−6.83, +3.91]. [Balanced cohort, figure, and paired results](MATH_ANSWER_TYPES.md)

### MuSiQue endpoint choice with earlier evidence

For each of 168 questions with two fixed candidates, a question counts as correct only if the model chooses the gold endpoint in both A/B orders. The given earlier passage determines the bridging entity. These results measure evidence-conditioned selection with supplied passages.

| Backbone and selected KL | Untouched | KAM | Dot | KAM minus dot (95% paired family interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, KL 4 | 5.36% | 4.96% | 3.97% | +0.99 [−0.25, +3.06] pp |
| Llama 3, KL 0.5 | 55.95% | 73.41% | 83.53% | −10.12 [−16.31, −3.65] pp |

A two-pass symmetric logit rule gave Llama 95.83% KAM and 95.44% dot on the same questions. With both possible earlier passages visible, Llama's four-order consistency was 8.93% untouched, 12.90% KAM and 17.86% dot. Combining the two endpoint margins at each early order gave 70.24%, 71.63% and 73.61% respectively. [Evidence swaps, label-order analysis and numeric records](PRETRAINED_ENDPOINT_CHOICE.md)

An explicit first-hop decision followed by endpoint choice gave all-four consistency of 51.79% untouched, 66.87% KAM and 72.82% dot. The architecture gap changed by −0.99 points [−10.98, +10.81] between direct and sequential choice. [Sequential composition and prompt work](PRETRAINED_ENDPOINT_CHOICE.md#separate-the-two-hops)

### TabMWP with a labeled archive table

On 128 grade-five-through-eight numerical questions, add a same-topic table from a different exercise and label it archival. The same three Llama seeds, KL 0.5, and neutral equivalence scorer apply in all conditions.

| Context | Untouched | KAM | Dot |
| --- | ---: | ---: | ---: |
| Current table only | 62.50% | 64.06% | 65.10% |
| Archive before current | 48.44% | 45.83% | 43.49% |
| Archive after current | 38.28% | 38.02% | 36.46% |

The archive-last condition changed the KAM–dot gap by +2.60 points [−4.80, +10.57] relative to current-only. [TabMWP archive noise, rates and paired intervals](TABMWP_ARCHIVE_NOISE.md)

On the 94-question matched subset at exact input length, replacing the same-topic archive with a different-topic archive raised KAM from 46.10% to 74.11% and dot from 43.62% to 66.31%. The paired difference between these two gains spans zero. [Exact-length control](TABMWP_ARCHIVE_NOISE.md#exact-length-different-topic-control)

On 128 fresh questions from previously unused title/header groups, typed current-table selection scored 50.78% KAM and 49.48% dot. Direct reading with the archive last scored 27.86% and 24.48%. The typed route used the correct caller-supplied record ID. [Overnight TabMWP follow-up](OVERNIGHT_ROUTING.md#fresh-tabmwp-results)

An exact-length replication on 83 of those questions raised KAM from 38.55% with a same-topic archive to 53.82% with a different-topic archive. Dot rose from 34.14% to 55.02%. The paired interval for the difference between their gains spans zero. [Fresh exact-length replication](OVERNIGHT_ROUTING.md#fresh-exact-length-replication)

### Live MuSiQue and release-clock routing

On 128 fresh questions from the publisher's MuSiQue training stratum, online sequential selection reached 65.36% KAM and 72.92% dot across all four placements. Batched mirrored selection reached 76.56% and 74.48%, with median isolated latency of 56.25 ms and 55.99 ms. The fitted adapters and their KL weights remained fixed. [Scope and deadline sweep](OVERNIGHT_ROUTING.md#live-musique-results)

On the same questions, BM25 first-hop selection followed by one endpoint call solved 68.23% KAM and 75.78% dot across all four placements at 60 ms per decision. In that paired rerun, direct solved 15.10% and 19.01% at 60 ms. The lexical route's measured median latency was 31.23 and 30.48 ms, including CPU first-hop ranking over the two supplied candidates. [Lexical routing](OVERNIGHT_ROUTING.md#one-pass-lexical-routing-follow-up)

With BM25 first-hop selection and mirrored endpoint scoring, all-four accuracy reached 93.75% KAM, 94.53% dot and 95.31% untouched. The same routes gave 78.39%, 82.55% and 89.84% within 60 ms per presentation. The BM25 single-view route was strongest at 40 ms. [Same-device deadline comparison](OVERNIGHT_ROUTING.md#mirroring-with-a-shorter-context)

With up to 20 native first-hop candidates, BM25 top one scored 40.36% KAM and 47.40% dot on verified first-hop and endpoint recovery across all four presentations within 60 ms each. A model call reranking the BM25 top two reduced deadline-constrained joint scores to 19.79% and 25.78%. [Native pool comparison](OVERNIGHT_ROUTING.md#first-hop-recall-with-native-distractors)

For these fixed A/B decisions, slicing the vocabulary projection preserved all 5,376 tested results across L4 and RTX PRO 6000 within device. The two-view median saving was 6.77–10.04 ms on L4 and 1.34–1.39 ms on RTX PRO 6000. [Hardware and parity comparison](OVERNIGHT_ROUTING.md#vocabulary-projection-for-fixed-choice-routing)

### Cached retrieval on fresh native MuSiQue pools

On 88 new questions from the publisher's training split, every first-hop pool contained 20 candidates. First-hop R@1 requires both pool orders. Joint recovery requires both verified passages across all four presentations within 60 ms each. Passage embeddings were cached, and query encoding was measured live on HPG.

| First-hop retriever | R@1 | KAM joint ≤60 ms | Dot joint ≤60 ms | Untouched Llama joint ≤60 ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 69.32% | 44.70% | 52.65% | 30.68% |
| Cached dense | 88.64% | 52.27% | 64.02% | 36.36% |
| Hybrid | 81.82% | 50.00% | 60.61% | 34.09% |

The same cached dense retriever gave KAM a gap of −11.74 percentage points versus dot [−18.38, −4.58] and +15.91 versus untouched [+9.52, +23.40] on this stratum. Hybrid led at R@3, 97.73% versus 95.45% for dense. The 12-question development split is reported separately. [Full comparison, timings and paired results](MUSIQUE_CACHED_RETRIEVAL.md)

### Routing capacity and nonlinear transfer controls

With the verified first passage supplied on 96 new MuSiQue relation families, native KAM recovered the endpoint in both orders on 68.40%, and reading the full KAM bank also gave 68.40%. Explicit dense dot reached 74.31%, and limiting dot to four live keys gave 71.88%. Their paired difference was −2.43 points [−6.94, +2.08]. Untouched Llama reached 55.21%. In a separate fixed-E5 first-hop probe, frozen E5, an added linear query map and a matched SiLU map gave 76.04%, 70.49% and 70.83% R@1 on 96 unseen families at the largest fit budget. [Controlled probes and learning curves](ROUTING_BOTTLENECKS.md)

### Matched live-token training

At 1.05 million tokens per run, the four live-token Llama adapters were evaluated on 98,304 PG-19 target tokens and 192 fresh HotpotQA questions. Endpoint retrieval requires correctness in both A/B orders.

| Variant | PG-19 perplexity ↓ | HotpotQA endpoint accuracy ↑ |
| --- | ---: | ---: |
| Untouched Llama | 15.3979 | 42.71% |
| Soft linear | 13.6623 | 66.84% |
| Soft SiLU | 13.6628 | 67.71% |
| Top-four linear | 13.6267 | 67.71% |
| Top-four SiLU | 13.6269 | 68.06% |

Top-four selection reduced PG-19 NLL by 0.00262 nats/token [−0.00339, −0.00187] and changed retrieval by +0.61 percentage points [−1.48, +2.86], averaging over activations. SiLU supplied no material gain at this budget. All arms used a dense equal-work diagnostic implementation. [Protocol, paired effects and ICML decision](LIVE_TOKEN_FACTORIAL.md)

### Long-context top-four confirmation

Reuse the three matched linear checkpoints at 4,096-token context on 48 new PG-19 books (147,456 scored targets) and a 320-question fixed-pool MuSiQue holdout.

| Variant | PG-19 perplexity ↓ | MuSiQue endpoint correct in both orders ↑ | Mirrored-margin accuracy ↑ |
| --- | ---: | ---: | ---: |
| Untouched Llama | 11.1776 | 1.25% | 65.94% |
| Soft linear | 10.2550 | 4.90% | 68.02% |
| Top-four linear | 10.2098 | 8.54% | 68.44% |

The perplexity difference was −0.44% under both parity-gated and common-reference scoring. The exploratory mirrored-margin retrieval gap was +0.42 points [−1.22, +2.05] under parity-gated scoring and +1.25 [−0.20, +2.76] under the common scorer. At 4,096 tokens, full-reader p50 was 193.32 ms with Flash and 193.58 ms with exact sparse value reads, with 469.14 MiB of incremental peak allocation in both cases. [Backend parity, common-scorer control and ICML decision](LONG_SPARSE_CONFIRMATION.md)

### Shortlisting the top-four read

On the same 4,096-token holdout and three top-four checkpoints, eight blocks scored 22.09% of eligible token keys. Exact-key recall was 72.25%, and 40.97% of sampled head queries recovered all four. Perplexity was 10.2083 versus 10.2098 for exact top-four. Mirrored MuSiQue was 68.33% versus 68.44%. Including proxy construction and lookup, the eight-block kernel took 87.50 µs versus 49.63 µs for Flash at 4,096 tokens. It beat Flash in isolation at 8,192 tokens. Batch-one full-reader latency and allocation offered no improvement. [All three budgets, peak memory and paired comparisons](BLOCK_SHORTLIST.md)

### Replacing attention in the Llama backbone

On the same holdout with frozen weights, full-model 4K prefill fell from 196.59 ms with Flash to 194.29, 192.14 and 187.90 ms with 8, 16 and 32 layers shortlisting eight blocks. Peak allocation was 469.14 MiB throughout. PG-19 perplexity rose from 10.21 to 33.76, 174.35 and 337.74. Mirrored MuSiQue fell from 68.44% to 48.23%, 52.50% and 50.31%. At 8K, all-layer replacement cut median prefill by 9.20%. [Paired quality, kernel validation and latency](BACKBONE_BLOCK_ATTENTION.md)

### Protecting the first block

Reserving the first block inside the eight-slot shortlist raised retained dense mass from 53.19% to 93.92% on prunable queries. On the same holdout, protected depths of 8, 16 and 32 gave PG-19 perplexity of 10.49, 10.80 and 11.20, compared with Flash at 10.21. Mirrored MuSiQue was 65.31%, 59.58% and 53.33%, compared with Flash at 68.44%. In the interleaved 4K profile, protected prefill cut p50 by 1.20%, 2.23% and 4.38%, with 469.14 MiB of incremental peak allocation throughout. [Layerwise mass, paired estimates and full timing](FIRST_BLOCK_ATTENTION.md)

### Distant evidence and selective Flash

At the final choice query, protecting the first block kept 14.98% of dense attention mass on earlier evidence beyond that block. The development-selected sixteen-layer Flash mask kept 75.43%, compared with 57.49% for the periodic mask. At the same 189.27 ms 4K median prefill, selected layers improved perplexity from 10.80 to 10.48. Mirrored MuSiQue was 59.38% selected and 59.58% periodic, a −0.21-point paired gap [−4.60, +4.22]. Flash scored 10.21 and 68.44% at 193.63 ms. [Layer-head audit and matched comparison](DISTANT_EVIDENCE_FALLBACK.md)

### Flash only near the end of the prompt

On the same holdout, Flash on the final 256 queries per layer reached 61.67% mirrored retrieval versus 54.27% when Flash was applied to the preceding 256 queries. The paired difference was +7.40 points [+2.50, +12.17]. Perplexity was 11.00 versus 10.98. The final window took 189.05 ms at 4K and 411.56 ms at 8K, versus 194.99 and 447.59 ms for Flash. [Matched query placement and latency](QUERY_WINDOW_FLASH.md)

On 128 fresh questions, final-256 gave 66.41% mirrored retrieval at 4K and 44.53% at 8K, compared with 59.64% and 50.78% for protected shortlisting and 74.48% and 55.47% for Flash. On the paired 23-book PG-19 panel, final-256 perplexity was 11.93 at 4K and 11.97 at 8K. It saved 3.10% and 8.08% of measured prefill time relative to Flash. [Independent transfer and paired length effects](QUERY_WINDOW_TRANSFER.md)

On 269 questions, replacing the annotated earlier text reduced signed mirrored margin by 0.442 logits for Flash, 0.127 for the final window, and 0.016 for protected shortlisting, relative to the same replacement in nearby background. The final-window improvement in this sensitivity over protected shortlisting was 0.111 logits [+0.054, +0.167]. [Controlled evidence use](DISTANT_EVIDENCE_ABLATION.md)

All 18 CUDA graph captures reproduced eager output exactly, including on changed input. The largest measured capture speedup was 0.88% across matched batches of one, two and four at 4K and 8K. [Matched graph timing and allocated memory](QUERY_WINDOW_GRAPHS.md)

On 128 fresh HotpotQA questions, final-256 reached 49.48% mirrored retrieval at 4K versus 62.24% for Flash, a −12.76-point paired gap [−20.05, −5.47]. At 8K the paired gap was +4.43 points [−4.69, +13.54]. Prefill was 3.02% and 8.11% faster at 4K and 8K. [Linked and unlinked cross-corpus results](LONG_HOTPOT_TRANSFER.md)

### Fixed block budget and routing coverage

Four prefix pins covered every audited MuSiQue bridge mention. Relative to four matched control pins, mirrored retrieval improved by 0.78 points at 4K [−5.78, +7.49] and 0.52 at 8K [−4.23, +5.43]. Both four-pin policies increased perplexity over ordinary first-block protection. [Prefix coverage and matched scores](FIXED_PREFIX_TRANSFER.md)

On disjoint PG-19 development books, the best small proxy correction retained 91.570% of dense mass versus 91.436% for the existing protected shortlist. An exact ranking at the same fixed eight-block budget reached 92.190%. [Mass retention ceiling and per-head results](BLOCK_ROUTER_DISTILLATION.md)

Computing only causally reachable proxy tiles preserved tested routes and outputs bitwise. Multiple launches almost doubled isolated batch-one route construction at 4K, and full-model prefill changed by less than 0.17% at all tested shapes. [Kernel and full-model timings](CAUSAL_PROXY_TILES.md)

Forcing the marked earlier passage into the same shortlist budget raised 4K gold-endpoint margins over matched background pins on MuSiQue by 0.139 logits and HotpotQA by 0.354. Unlinked mirrored accuracy improved by 10.97 and 10.42 points. Linked and 8K accuracy differences included zero. [Matched evidence-block intervention](ANNOTATED_BLOCK_ROUTING.md)

With a fixed four/twelve block-budget histogram, selecting layers by dense output reconstruction reached 59.64% on 8K HotpotQA and 55.21% on 8K MuSiQue, compared with 42.71% and 55.47% for Flash. Prefill was 9.21% faster at 8K, and PG-19 perplexity rose from 10.81 to 12.60. At 4K, the same schedule reduced MuSiQue accuracy versus uniform eight by 10.94 points. [Layer budgets, latency and paired quality](LAYER_BLOCK_BUDGET.md)

Bitwise-gated launch tuning reduced isolated protected attention CUDA time by 12.70–18.02% on the RTX PRO 6000. Full-reader prefill saved 0.49–0.63% over the same protected policy across 4K/8K and batches one/four. [Tuned kernel and full-reader cost](SPARSE_KERNEL_TUNING.md)

### Native long-document comprehension

On 452 QuALITY questions, final-window Flash reached 61.43% original-order accuracy and 66.59% after balancing four option positions, compared with 60.91% and 69.10% for Flash. Protected shortlisting and the reconstruction-selected layer schedule reached 58.04% and 58.78% rotation-balanced accuracy. At 8,189 tokens the final window used 415.91 ms versus 453.37 ms with Flash. [Intact articles, answer calibration and timing](NATIVE_LONG_DOCUMENT_QA.md)

On held-out QuALITY articles, the fitted two-pass margin gate deferred 1.34% of final-window predictions and recovered 0.45 points over its first pass, at 286.31 ms modeled prefill versus 301.19 ms for Flash. The other confidence gates retained a larger accuracy gap. [Predicted two-pass costs and oracle ceiling](CONFIDENCE_FLASH_DEFERRAL.md)

On a five-fold article-held-out replay of the same questions, a learned final-window rescue ranker reran 8.63% of cases and gained eight correct answers while breaking eight. The margin rule gained eight with 1.70% reruns. Both use first-pass logits and modeled timing. [Cost-aware learned rescue ranking](QUALITY_RESCUE_RANKING.md)

### Frozen attention reconstruction and sparse training checks

On 13 leave-one-book-out calibration folds, a four/eight/twelve layer allocation reduced relative squared output error from 4.755% to 4.444% at the same nominal block budget. Mass gain and output-error reduction had layerwise Pearson correlation −0.01. These are frozen-activation reconstruction metrics. [Derivation, counterexample and calibration replay](ATTENTION_MASS_AND_RECONSTRUCTION.md)

On a separate 16-book SmolLM2 panel, omitted eight-token mean summaries reduced frozen projected error by 10.71%. Live 4K perplexity was 13.58 for dense, 15.14 for protected shortlisting, and 15.07 with the mean summaries. The paired mean-summary NLL interval included zero. [Pretrained SmolLM2 reconstruction and live replay](OMITTED_BLOCK_SUMMARIES.md)

The same pretrained model assigned at least 95% of dense attention mass to the protected shortlist on 39.40% of sampled rows. The compact mathematical bound certified zero of 967,680 samples even when evaluated separately per query. [Certificate on real attention states](PRETRAINED_MASS_CERTIFICATE.md)

On eight calibration-held-out PG-19 books, the restricted dense head-group selection lowered projected error by 23.00% and live perplexity to 13.91 from 13.99 for the same-budget calibrated whole layers. Unrestricted head groups reached 13.82 and full dense reached 13.46. The prototype used Apple M5, with a separate per-presentation latency control. [Head scope, perplexity and prefill](HEAD_SELECTIVE_FLASH.md)

The fraction of MuSiQue predictions where Flash led by at most half a logit rose from 44.79% at 4K to 86.46% at 8K. The final window changed 20.05% and 51.04% of decisions at those widths, even as its median centered logit error fell. On intact QuALITY it changed 3 of 689 decisions with a Flash lead above two logits. [Decision stability across corpora](DECISION_MARGIN_STABILITY.md)

Six matched adapter continuation jobs completed at 9,068–9,079 input tokens/s for Flash and 7,387–7,615 tokens/s for differentiable sparse attention. HPG canceled the quality evaluation array. [Training fidelity and receipts](SPARSE_ADAPTER_RECOVERY.md)

### Release-clock follow-up

On 61 public release snapshots where GitHub and PyPI timestamps changed the correct version, neither fitted arm solved both clocks in all four answer placements. Clock eligibility constraints raised this rate to 59.02% KAM and 53.55% dot. [Release-clock evaluation](OVERNIGHT_ROUTING.md#public-release-clocks)

### Held-out next-token perplexity

Each cell covers 32,768 held-out tokens from a fixed source stream. KAM and dot use the high-rate KL recipe and the same three seeds. The Wikipedia and Python rows use bounded samples of the released WikiText-103 and CodeSearchNet test splits. Arithmetic uses the fixed synthetic templates with an independent test seed. Tokenizers differ across backbones.

| Backbone and source | Untouched | KAM | Dot |
| --- | ---: | ---: | ---: |
| SmolLM2, WikiText-103 | 11.58 | 10.96 | 10.46 |
| SmolLM2, CodeSearchNet Python | 4.60 | 4.45 | 4.31 |
| SmolLM2, synthetic arithmetic | 3.79 | 2.56 | 2.10 |
| Llama 3, WikiText-103 | 10.17 | 8.88 | 8.74 |
| Llama 3, CodeSearchNet Python | 5.00 | 4.10 | 4.04 |
| Llama 3, synthetic arithmetic | 6.00 | 2.26 | 2.20 |

[All recipes, matched-drift controls, and efficiency measurements](GENERAL_BENCHMARKS_DIAGNOSTICS.md)

### Preserving pretrained scores on HiPerGator

SmolLM2-1.7B-Instruct is the shared backbone and untouched reference. KAM and dot each add 1,114,112 trainable parameters. Every cell reports paired means from three seeds in KAM / dot order.

| Benchmark | Untouched | Reference KAM / dot | Lower LR KAM / dot | Lower LR + KL KAM / dot |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 28.66% | 30.89% / 29.07% | 28.05% / 27.03% | 27.64% / 28.46% |
| MBPP | 38.00% | 36.87% / 32.33% | 37.20% / 33.67% | 38.13% / 37.07% |
| GPQA Diamond | 25.76% | 24.92% / 24.41% | 26.77% / 23.74% | 26.26% / 23.06% |
| MMLU | 42.50% | 38.45% / 39.16% | 42.50% / 36.51% | 42.54% / 42.09% |
| GSM8K | 31.77% | 12.31% / 6.92% | 30.00% / 3.59% | 30.50% / 27.29% |

[Preservation report and full protocol](GENERAL_BENCHMARKS_PRESERVATION.md) · [Numeric summary](../../results/general_benchmarks_preservation/summary.json)

### Shared pretrained backbone

UF Research Computing prohibits Qwen use on HiPerGator. These historical scores are retained for audit. [UF policy](https://docs.rc.ufl.edu/training/new_user_training/#prohibition-on-large-language-models-from-countries-of-concern)

Both adapters start from the same frozen Qwen2.5-1.5B-Instruct weights. Each adds 851,968 trainable parameters and trains on the same 2,097,152 subword tokens. The untouched checkpoint uses the same Qwen prompts and scoring.

| Benchmark | Metric | Untouched Qwen | KAM adapter | Dot adapter |
| --- | --- | ---: | ---: | ---: |
| HumanEval | Pass@1 | 50.61% (83/164) | 47.56% (78/164) | 42.07% (69/164) |
| MBPP | Pass@1 | 48.20% (241/500) | 33.40% (167/500) | 41.80% (209/500) |
| GPQA Diamond | Accuracy | 30.30% (60/198) | 23.74% (47/198) | 29.29% (58/198) |
| MMLU | Accuracy | 57.75% (8,109/14,042) | 54.45% (7,646/14,042) | 55.34% (7,771/14,042) |
| GSM8K | Strict exact match | 21.99% (290/1,319) | 17.29% (228/1,319) | 16.45% (217/1,319) |

[Pretrained adapter report and protocol](GENERAL_BENCHMARKS_PRETRAINED.md) · [Numeric summary](../../results/general_benchmarks_pretrained/summary.json)

## Natural QA and conversational memory

**Answer EM** is the fraction of normalized answers that exactly match a reference. **Token F1** measures partial overlap with a reference answer. **Complete evidence** requires every support unit at the study's documented granularity. **Joint EM** here requires both answer EM and complete evidence. Official benchmark joint scores can use different support units and formulas. Counts below are baseline → comparison on the same cohort. A dash marks a metric without a comparable value in the scorecard.

| Dataset and slice | Matched comparison | Answer EM / accuracy | Token F1 | Complete evidence | Joint EM |
| --- | --- | ---: | ---: | ---: | ---: |
| [HotpotQA](../../reports/nonlinear_retrieval_natural/CONDITIONED_UNION.md), 1,024 fresh bridge questions | Broad top 64 → conditioned union, same reranker and FLAN-T5-small | **396 → 422** (38.67% → 41.21%) | — | 481 → 511 | 308 → 333 (30.08% → 32.52%) |
| HotpotQA, originally complete (640) | Same comparison | 351 → 347 | — | 481 → 465 | 308 → 298 |
| HotpotQA, originally incomplete (384) | Same comparison | 45 → 75 | — | 0 → 46 | 0 → 35 |
| [2WikiMultiHopQA](../../reports/nonlinear_retrieval_natural/NATIVE_TWOWIKI_RERANK.md), 1,024 native questions | Dense 128 → conditioned 128, frozen HotpotQA heads and reader | **217 → 238** (21.19% → 23.24%) | — | 320 → 383 | 149 → 165 |
| [MuSiQue](../../reports/nonlinear_retrieval_natural/MUSIQUE_SUBQUERY_ASSIGNMENT.md), 576 questions, three seeds | Direct rank → distinct assignment, same four-passage budget and reader | **48–56 → 70–73** (8.33%–9.72% → 12.15%–12.67%) | — | 61–68 → 84–87 | 22–25 → 38–42 |
| [IIRC](../../reports/nonlinear_retrieval_natural/IIRC_READER_FORMAT.md), 134 aligned questions | Generic → question-derived answer format with gold article leads supplied | **25 → 42** (18.66% → 31.34%) | — | — | — |
| [MultiHop-RAG](../../reports/nonlinear_retrieval_natural/NEWS_ARTICLE_WINDOWS.md), 384 news questions | Word-stride → one sentence center per article, annotated articles supplied | **111 → 134** (28.91% → 34.90%) | — | 46 → 69 | 16 → 31 |
| [QASPER](../../reports/nonlinear_retrieval_natural/QASPER_BINARY.md), 74 binary questions | Always yes → dense top-two paragraphs in supplied paper | **48 → 47** (64.86% → 63.51%) | — | — → 26 | — → 15 |
| [TimeQA](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#timeqa-prose-and-same-entity-date-collisions), 640 held-out entity contrasts | Dated dense → learned date-aware ranking | **80 → 184** (12.50% → 28.75%) | — | 117 → 261 | 49 → 126 |
| [TempReason](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#tempreason-fact-text), absolute, 384 noisy cases | Dense top two → time and entity rules, same reader | **147 → 358** (38.28% → 93.23%) | — | 320 → 384¹ | — |
| TempReason, relative, 384 noisy cases | Same comparison | **110 → 327** (28.65% → 85.16%) | — | 133 → 380¹ | — |
| [LongMemEval-M](../../reports/nonlinear_retrieval_natural/LONGMEMEVAL_READER_BUDGET.md), 470 answerable | Dense prefixes → query windows, fixed 512-token budget | **28 → 41** (5.96% → 8.72%) | **9.97% → 13.56%** | — | 23 → 36 |
| [TIME-Lite](../../reports/nonlinear_retrieval_natural/TIME_LITE_CONTEXT.md), dialogue, 220 | No context → dense windows, FLAN-T5-base | **63 → 85** (28.64% → 38.64%) | — | — | — |
| TIME-Lite, news, 210 | Same comparison | **84 → 140** (40.00% → 66.67%) | — | — | — |
| TIME-Lite, wiki, 66 | Same comparison | **23 → 26** (34.85% → 39.39%) | — | — | — |

¹ TempReason evidence means the selected facts contain an accepted answer string. HotpotQA answer EM was totaled from the exactness flags in the [archived replay](../../results/nonlinear_retrieval_natural/conditioned_union_v1/replay.json.gz). LongMemEval-M F1 comes from its [numerical summary](../../results/nonlinear_retrieval_natural/longmemeval_m_reader_budget_v1/summary.json). MuSiQue ranges are across seeds. Its four-hop joint score was **0/160** for each seed in this comparison. HotpotQA joint EM gained **2.44 percentage points** (95% paired interval **+1.17 to +3.81**) overall and fell **1.56 points** (**−2.66 to −0.63**) on complete pools. Filling the missing QA F1 and official support metrics requires a scorer replay with the corresponding answer text and support units.

## Retrieval and structured memory

Hit@k asks whether a target appears in the first k results. Complete recall@k asks whether every required target is present. Recall@k measures the fraction of relevant targets recovered, and nDCG@k also rewards placing them near the top. Structured exact accuracy checks a discrete answer, such as a count or snapshot.

| Dataset | Metric and comparison | Matched result |
| --- | --- | --- |
| [BRIGHT](../../reports/nonlinear_retrieval_natural/BRIGHT_PASSAGE_WINDOWS.md), economics / earth science / psychology / sustainable living (428 total) | Hybrid nDCG@10 on a 0–100 scale, head → multiple windows | 15.79 → 19.05 / 17.79 → 23.01 / 10.61 → 15.01 / 12.19 → 13.48 |
| BRIGHT, same domains | Recall@64, same comparison | 32.70% → 35.62% / 35.81% → 42.20% / 20.81% → 27.95% / 30.52% → 34.05% |
| [IIRC](../../reports/nonlinear_retrieval_natural/IIRC_MATCHED_SEQUENTIAL.md), 153 | Complete recall@2, sequential → joint scoring | Neutral 62 → 66, title-aware 65 → 73 |
| [LongMemEval-S](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#conversational-memory-and-annotation-clocks), 470 answerable | Complete session recall@5, BM25 → E5 | 390 → 430 |
| [LongMemEval-M](../../reports/nonlinear_retrieval_natural/LONGMEMEVAL_M.md), 470 answerable | Complete session recall@5, nested 64 sessions → full | 423 → 350 |
| [NoLiMa-derived probe](../../reports/nonlinear_retrieval_natural/CROSS_DOMAIN_TRANSFER.md#controlled-reduction-in-lexical-overlap), depth 64 | Dense hit@2, clean → lexical decoy | One-hop 9 → 6/56, two-hop 6 → 3/44. Clean direct control 56/56. |
| [TimelineQA atomic](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#seeded-lifelogs-and-conflicting-provenance), 576 generated cases | Dense hit@1 at depth 256, authentic → false displayed date | 438 → 312 |
| [TimelineQA counts](../../reports/nonlinear_retrieval_natural/TIMELINEQA_COUNTS.md), 380 generated cases | Exact count with false dates, displayed → trusted source month | 0 → 380. Dense complete recall@8 under false dates was 97/288 among positive cases. |
| [BABILong](../../reports/nonlinear_retrieval_natural/BABILONG_REPLAY.md), 64k context, eight windows | Exact denotation plus nonempty witness, E5 → grammar-filtered E5 | QA2 1 → 41/100, QA3 0 → 3/86, QA7 45 → 91/100, QA8 64 → 95/100. Complete structured replay agreed on 1,200/1,200 across tasks and lengths. |
| [UnSeenTimeQA](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#executable-labels-for-fictional-event-schedules) | Exact answer-set agreement, deterministic schedule replay | 3,600/3,600 easy/medium serial, 2,693/2,700 hard serial, 2,618/2,700 hard parallel from text-derived dependencies |
| [UnSeenTimeQA duration swaps](../../reports/nonlinear_retrieval_natural/DURATION_POINTER.md), 1,173 pairs | Both event pointers exact, learned dot / radial given typed inputs | 52.77% / 49.53% |
| [Philadelphia Fed vintages](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#real-revision-and-abstention) | Exact release, dense → semantic identity plus exact date guard | Known canonical 208 → 320/320, known paraphrases 130 → 80/320. Unavailable canonical and paraphrase each 0 → 160/160 correct abstentions. |
| [Dallas Fed OECD GDP](../../reports/nonlinear_retrieval_natural/TEMPORAL_RELATIVE_CLOCK.md), 584 questions × two recodings × three seeds | Exact address and value, radial signed log → relative mean signed log | 42.44% → 57.56% at depth 48 |
| [Eurostat unemployment and industrial production](../../reports/nonlinear_retrieval_natural/ADDRESS_LOCAL_CLOCK.md) | Strict archived snapshot accuracy, mixed dot clock → clock plus within-address fact rank | Unemployment 3.01% → 81.37%, industrial production 0.20% → 91.35%, with typed archive inputs |
| [Eurostat text routing](../../reports/nonlinear_retrieval_natural/EUROSTAT_TEXT_ROUTING.md), 864 answerable authored prompts | Exact snapshot, dense → dense address plus parsed month and exact cutoff | 114 → 864. Separate gates answered 648/864 and refused all 324 tested unsupported or ambiguous prompts. |
| Project revision memory | Selective correctness under held-out aliases | Conservative registry gate answered 48/192 supported prompts and refused 576/576 unseen-identity prompts. [Registry results](../../results/nonlinear_retrieval_natural/release_registry_sample_v1/summary.json) |

PG-19 supplied the BABILong filler and Wikipedia supplied the indexed QA corpus. Their reported scores apply to the downstream tasks. The [dataset portfolio](DATASET_PORTFOLIO.md) also lists prospective datasets, including RULER.

## Direct model and controlled tasks

| Dataset or task | Conventional metric | Archived result |
| --- | --- | --- |
| Tiny Shakespeare | Character NLL, perplexity and bits per character, lower is better | Affine C0 / nonlinear C3 / parameter-matched linear branch at 1,000 updates, three development seeds: **1.5946 / 1.5906 / 1.5891 nats per character**. Exponentiating mean NLL gives **4.93 / 4.91 / 4.90** character perplexity. Dividing mean NLL by ln 2 gives **2.300 / 2.295 / 2.293** bits per character. The archived M4 text study reserved the test split. |
| [Bundled small-language fixture](../../reports/phase6/PHASE6_TRANSFORMER_COMPARISON.md) | Loss and perplexity | The 256-token operational profile has no comparable held-out score. |
| [MQAR, variable copy, Dyck-2, reusable regime grammar](../../reports/phase2/PHASE2_RESULTS.md#five-seed-mechanism-language-screen) | Token accuracy and cross-entropy, plus sequence accuracy where recorded | Five-seed architecture tables. Separate length holdouts found 8.26% token accuracy and 0% exact sequence at copy length 80, after training on 8–64. Dyck-2 at depth 16 yielded 42.14% token accuracy and 0% valid outputs. |
| [Short copy and character smoke fixtures](../../reports/phase3/PHASE2_EVIDENCE_AUDIT.md) | Checkpoint inventory | Legacy artifacts catalogued. No common held-out score established. |
| [Mackey–Glass and NARMA, stationary and switching](../../reports/phase2/PHASE2_RESULTS.md#paired-stationary-and-switching-evidence) | Held-out MSE, with normalized error in adaptation studies | Five-seed comparisons for all four tasks. No corrected paired improvement passed the reported significance gate. |
| [Prototype switches and controlled regression](../../reports/phase4/PHASE4_FACTORIAL_REPORT.md) | Held-out MSE by task and regime | Factorial results and [transformer screen](../../reports/phase6/PHASE6_STAGE1_FULL_REPORT.md) |
| [Procedural bitemporal revision](../../reports/nonlinear_retrieval_natural/GUARDED_CURRICULUM_ORDER.md) | Exact alias and payload at unseen depth 1,024 | Guarded radial, cyclic long → shallow-first, 11 → 1,535/1,536 at equal tokens and updates |
| [Fixed-key and composed address probes](../../EXPERIMENTS.md) | Exact address and payload | Generated lookup, context and positional controls |

Generalization is the score on a specified held-out condition, such as new articles, entities, countries or sequence lengths. The linked studies report how the held-out sets were formed and how uncertainty was estimated.

## Efficiency at measured quality

| Workload | Efficiency metric | Quality check |
| --- | --- | --- |
| [Guarded radial inference](../../reports/nonlinear_retrieval_natural/GUARDED_FP32_SDPA.md), L4, depth 1,024 | Eager → fused FP32 plus crop, **39.28 → 1.79 ms** per measured batch | Tested hard decisions matched across L4, RTX PRO 6000 and B200 |
| [HotpotQA full requests](../../reports/nonlinear_retrieval_natural/FULLWIKI_SENTENCE_THROUGHPUT.md#confirmation-on-800-different-requests), L4 | Single FP32 → eight requests with FP16 sentence encoding, **10.61 → 19.83 requests/s** | Support, answer EM and joint flags matched on all 800 |
| [Radial training](../../reports/nonlinear_retrieval_natural/RADIAL_CURRICULUM_CAPTURE.md), B200, 19,200 updates | Eager → CUDA Graph, **159.38 → 30.04 s** excluding setup | Complete optimizer trajectories matched under tested deterministic settings |

Latency p50 and p95 describe typical and tail requests. Throughput measures completed requests or processed tokens per second. Compare at the same workload, hardware, context length and quality target. [Serving and training details](RESULTS.md)
