# Chronological presentation of fixed LongMemEval turns

Chronological prompt order produced no detectable gain in strict temporal questions for either frozen reader. On 127 temporal-reasoning cases, FLAN-T5-large reached eight joint exact matches in both ascending chronological order and the matched timestamped retrieval order. These were the same eight individual questions.

The experiment freezes the retrieved or marked turn set and the token slice retained from every turn. All dense prompts use the same 1,024-token query windows as the reader-size study. The marked-turn arms keep their previous 1,024-token prefixes. The timestamped arms reorder packed chunks without changing text, dates, truncation, evidence membership or model weights. Ascending and descending policies preserve the turn order within a session. A deterministic case-specific shuffle controls for reordering itself. All 2,000 previously evaluated undated reader records replayed exactly.

| Dense prompt order | Base joint, all 470 | Large joint, all 470 | Base temporal joint, 127 | Large temporal joint, 127 |
| --- | ---: | ---: | ---: | ---: |
| Retrieval order, undated | 25 | 52 | 8 | 8 |
| Retrieval order, dated | 21 | 58 | 8 | 8 |
| Ascending, dated | 23 | 55 | 5 | 8 |
| Descending, dated | 25 | 55 | 8 | 6 |
| Shuffled, dated | 27 | 64 | 6 | 6 |

The larger reader gained six overall joint answers when dates were added without reordering, +1.28 percentage points (95% shared-evidence interval +0.21 to +2.55). Chronological versus dated retrieval order had intervals spanning zero for both readers overall. The timestamped shuffle obtained the largest overall count with the large reader, within paired uncertainty of ascending order. Among the 92 temporal questions with an eligible occurrence of every required session by the question date, the large reader reached four joint successes with retrieval order and four with ascending order. The other 35 are reported separately with their unmodified author labels. There is little evidence here that prompt ordering addresses the temporal gap.

HPG array 52993155. [Frozen times, ordering audits and paired outcomes](../../results/nonlinear_retrieval_natural/longmemeval_m_timeline_order_v1/) · [Reader scale control](LONGMEMEVAL_READER_SIZE.md)
