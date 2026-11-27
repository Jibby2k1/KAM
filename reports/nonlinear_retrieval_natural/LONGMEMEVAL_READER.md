# Reading selected turns and conversation dates on LongMemEval-M

On 470 answerable LongMemEval-M questions, a frozen FLAN-T5-base reader reached 28 normalized exact matches from the top-ranked 20 turns in five dense-ranked sessions. Providing marked turns from the full history raised that count to 121. Adding conversation dates to the same retained text produced only small changes overall.

## Matched reader controls

All deployments use the frozen session and turn records from the medium-history benchmark. The reader has a fixed 512-token context and may emit up to 20 answer tokens. Prompt construction divides the remaining token budget evenly across selected turns, redistributing unused slots. Undated and dated controls include exactly the same retained text tokens, with space for dates reserved in both. Dates are session and question timestamps as supplied by the dataset. They describe the conversation clock.

The marked-in-dense oracle prioritizes one annotated turn per available gold session and then other available marks within the five dense-ranked sessions, using the original dense list to fill remaining slots. The full-history oracle selects up to 20 marked turns without background filler. This changes both the set of accessible sessions and the amount of distracting or truncated text. The result measures diagnostic headroom for their combination.

| Condition | Complete marked-turn support | Normalized EM | Joint EM | Detected abstention |
| --- | ---: | ---: | ---: | ---: |
| No history | 0 | 13 | 0 | 438 |
| Dense sessions and turns | 326 | 28 | 23 | 210 |
| Dense plus dates | 326 | 31 | 27 | 153 |
| User-only session ranking | 330 | 30 | 26 | 201 |
| Marked turns within dense sessions | 342 | 31 | 27 | 206 |
| Full-history marked turns | 438 | 121 | 116 | 167 |
| Full-history marked turns plus dates | 438 | 122 | 119 | 152 |

Joint EM requires the normalized answer and at least one presented author-marked turn from each support session. A marked turn can still be truncated before the decisive words. Every dense prompt had at least one truncated turn, compared with 19 full-history marked prompts. Numbers in the source answer field were converted to strings. These scores use normalized string matches and token F1, with a short reader and its output limit held constant. The official LongMemEval judge may accept paraphrases or long-form answers that these scores miss.

The full-history oracle gained 19.15 percentage points of normalized EM over marked-in-dense, with a shared-evidence cluster interval of +15.20 to +22.88. At fixed selected text, dense plus dates changed EM by +0.64 points [−0.85, +2.14]. The knowledge-update subset moved from 4 to 8 exact dense answers out of 72 and from 28 to 34 with marked turns. Temporal reasoning stayed at 14 of 127 in both dense conditions and moved from 19 to 15 in the marked controls. On the 30 designated abstention items, the short reader produced a recognized abstention for 15 with dense evidence, seven with dated dense evidence, and 26 with no history. The limited exact-match results warrant a separate longer answer and context control before interpreting this as model-wide reader failure.

## Matched candidate enrichment

For six, eight and ten slots, a separate replay kept the dense top five, appended unique decomposition candidates, then used dense continuation to fill the remaining budget. The three bound decomposers recovered every gold session on 363–364, 384–386 and 394–395 of 470 questions. Dense alone at the same depths recovered 364, 386 and 396. Within the 127 temporal questions, bound enrichment reached 89–90, 93–95 and 95–96 versus dense 89, 94 and 97. Preserving the dense prefix removed the large top-five transfer penalty without delivering consistent enrichment over simply going deeper.

HPG array 52983463 evaluated the fixed reader arms on four L4 GPUs. [Reader measurements](../../results/nonlinear_retrieval_natural/longmemeval_m_reader_v1/) · [Equal-budget enrichment](../../results/nonlinear_retrieval_natural/longmemeval_m_transfer_v1/enrichment.json) · [Retrieval transfer](LONGMEMEVAL_TRANSFER.md)
