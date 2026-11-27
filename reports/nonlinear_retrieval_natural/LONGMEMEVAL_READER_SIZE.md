# Reader capacity at fixed LongMemEval prompts

On the same 470 answerable LongMemEval-M questions, replacing FLAN-T5-base with FLAN-T5-large raised joint exact match from 25 to 52 with 1,024-token query windows. On 127 temporal-reasoning cases, both readers reached eight joint exact matches in that condition.

Both readers receive identical prompt token IDs, sorted and batched in the same order. Dense prompts retain the same five sessions and 20 ordered turns as the prior study. Oracle prompts use author-marked turns, capped at 20. Decoding is greedy with 20 new tokens. The evaluation separately measures normalized answer exact match, exact answer where annotated turns from every required session were selected, and explicit abstention on 30 unavailable prompts. All 2,500 shared base prompt outcomes and packing records replayed exactly.

| Fixed evidence | Base answer / joint | Large answer / joint | Temporal joint base / large | Abstentions base / large |
| --- | ---: | ---: | ---: | ---: |
| Dense, 512-token prefix | 28 / 23 | 40 / 33 | 9 / 7 | 15 / 13 |
| Dense, 512-token query windows | 41 / 36 | 51 / 43 | 9 / 7 | 13 / 15 |
| Dense, 1,024-token query windows | 32 / 25 | 60 / 52 | 8 / 8 | 5 / 12 |
| Marked turns, 1,024-token prefix | 122 / 117 | 145 / 136 | 14 / 12 | 23 / 13 |
| No history | 13 / 0 | 22 / 0 | 0 / 0 | 26 / 16 |

At the 1,024-token dense window, the paired joint gain was +5.74 percentage points (95% shared-evidence cluster interval +3.19 to +8.32). The temporal slice gained and lost two cases each. Median inference time per shard across all six prompt arms was 15.85 seconds for base and 48.55 for large on L4. Larger capacity improved general reading on these prompts at about 3.1× this readout cost. The evidence and temporal outcome differences remain distinct. Abstention moved in both directions across the prompts.

HPG array 52990304. [Matched records and category labels](../../results/nonlinear_retrieval_natural/longmemeval_m_reader_size_v1/) · [Prompt budget diagnostic](LONGMEMEVAL_READER_BUDGET.md) · [Reader checkpoint](https://huggingface.co/google/flan-t5-large)

[Matched chronological prompts](LONGMEMEVAL_TIMELINE_ORDER.md) test the same retained turn text with and without reordered dates.
