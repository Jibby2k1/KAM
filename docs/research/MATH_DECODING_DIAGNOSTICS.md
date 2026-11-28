# Decoding length, repetition, and work on mathematical answers

The [neutral-prompt MATH-500 study](FULL_MATH500.md) uses fixed KL settings, matched checkpoints, and greedy EOS-or-cap generation. Llama KAM reached 1,024 tokens on 23.93% of items and dot on 14.07%. Longer generation can make the answer available, and it can also prolong an incomplete derivation. These diagnostics keep correctness and actual decoding work visible together.

## Independently replay the 384-token prefix

Four HPG RTX PRO 6000 jobs re-ran one original 64-item batch for both fitted arms on both backbones, using seed 731221. The generated text and length matched the recorded 384-token prefixes in all 256 cases. Each job alternated the decoding loop that bypasses content checks with the older loop configured with a pattern that never matches. Both loops returned the same text and length in all repeats.

| Backbone and arm | Bypass content checks (seconds) | Never-match content check (seconds) |
| --- | ---: | ---: |
| SmolLM2 KAM | 5.61 | 5.37 |
| SmolLM2 dot | 5.40 | 5.23 |
| Llama KAM | 13.61 | 13.49 |
| Llama dot | 13.44 | 13.36 |

These are medians of two timed passes per variant. The measured panel gives no evidence of a speed advantage from the bypass. Model load and queue wait sit outside the timing. [Replay and timing receipts](../../results/general_benchmarks_full_math500/prefix-parity.json) · [Scheduler receipts](../../results/general_benchmarks_full_math500/prefix-parity-scheduler.tsv) · [Probe](../../scripts/probe_kam_math500_prefix.py)

## Compare stopping policies on the same trajectories

An extractable expression at token 384 can be an intermediate result. A policy that continued only when the prefix grader found no candidate recovered almost none of the longer-run score on Llama. KAM remained at 27.33% and dot moved from 27.07% to 27.13%. It still required mean summed batch decoder steps of 7,125 for KAM and 6,922 for dot, compared with 3,072 for the fixed prefix and 8,192 for the full cap. This is a replay on existing outputs.

A second rule flags a prefix if at least 75% of the six-token windows in the last 144 lexical tokens repeat. On Llama, KAM averaged 50 capped outputs with high terminal repetition and dot averaged 25.3. At token 384, KAM averaged 40.3 flagged prefixes and dot 17.7. Every flagged prefix in this sample later hit the cap. Stopping these traces at 384 preserved all per-item correctness flags. The rule uses lexical tokens for its decision and model tokens for its generation limit.

| Llama adapter | Model token slots saved at batch 1 | At batch 8 | At batch 64 |
| --- | ---: | ---: | ---: |
| KAM | 13.02% | 8.25% | 2.19% |
| Dot | 6.99% | 9.32% | 1.76% |

Each group contributes its size times its longest required decoding length. The batch-one and batch-eight rows regroup recorded trajectories as a work proxy. The table measures model token slots and leaves measured runtime to the independent serving profile. [Llama repetition flags](../../results/general_benchmarks_full_math500/llama3-repetition.json) · [SmolLM2 repetition flags](../../results/general_benchmarks_full_math500/smollm2-repetition.json) · [Llama cost replay](../../results/general_benchmarks_full_math500/llama3-budget-policy.json) · [SmolLM2 cost replay](../../results/general_benchmarks_full_math500/smollm2-budget-policy.json)
