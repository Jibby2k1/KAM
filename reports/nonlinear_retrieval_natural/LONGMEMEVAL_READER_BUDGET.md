# Fixed LongMemEval evidence at different reader budgets

On the same 470 answerable LongMemEval-M questions, choosing a query-matching window within each of the 20 dense-ranked turns raised joint exact match from 23 to 36 at a 512-token context budget. Expanding the prompt to 1,024 or 1,536 tokens produced smaller or negative gains for the frozen FLAN-T5-base reader in this replay.

The dense arms use the exact same five sessions and 20 ordered turns. The fixed context budget is divided across turn contents with redistribution for short turns. The window policy picks the contiguous span of the allotted length containing the most distinct query subword tokens from words at least four characters long, breaking ties toward earlier spans. The prefix and window policies retain the same number of body tokens for every matched question and turn. The marked-turn oracle uses the same turn sets as the first reader experiment. Both 512-token prefix controls replayed all 500 previous records exactly.

| Dense evidence policy | Median body tokens | Answer EM | Joint EM | Temporal joint EM, out of 127 |
| --- | ---: | ---: | ---: | ---: |
| 512, prefix | 323 | 28 | 23 | 9 |
| 512, query window | 323 | 41 | 36 | 9 |
| 1,024, prefix | 835 | 29 | 25 | 5 |
| 1,024, query window | 835 | 32 | 25 | 8 |
| 1,536, prefix | 1,347 | 21 | 16 | 2 |

The 512-token window gained +2.77 percentage points of joint EM with a shared-evidence cluster interval of +0.42 to +5.12. The 1,536-token prefix lost 1.91 joint points compared with the 1,024-token prefix [−3.81, −0.21]. The window's gain left the strict temporal-reasoning joint count unchanged. With marked turns supplied, expanding 512 to 1,024 tokens moved EM from 121 to 122 and cut clipped prompts from 19 to one. Increasing the generation cap from 20 to 64 tokens at fixed 1,024 input tokens preserved every EM outcome for both dense and marked evidence. The longer generation cap did improve token F1 slightly.

The result applies to the frozen FLAN-T5-base reader and lexical answer metric. It separates packing from session and turn retrieval. More room in the prompt exposed more text, including distractors, so choosing the useful span remains a separate issue from retrieving a marked turn. HPG array 52984694 ran the lengths and evidence arms on four L4 GPUs. [Prompt and reader records](../../results/nonlinear_retrieval_natural/longmemeval_m_reader_budget_v1/) · [Fixed evidence and date diagnostics](LONGMEMEVAL_READER.md)
