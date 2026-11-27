# Relative session dates at a fixed LongMemEval evidence budget

Precomputing the time between each conversation and the question did not improve the larger reader's temporal joint exact match. At a fixed evidence budget, the calendar-only and calendar-plus-offset arms each reached eight of 127 strict temporal joint matches with dense retrieval, and 14 with author-marked turns.

The three prompts display an ISO session date, a signed day distance from the question date, or both. The question date appears in every arm. The marker describes the conversation date. The event described in the conversation can have its own date. Fixed positions and longest-label budgeting preserve the identical retained turn tokens and order across the three encodings within each evidence arm. The effective text budget in this experiment is a little smaller than in the earlier chronology trial because it accommodates the longest marker. Total input is at most 1,024 tokens, output at most 20 tokens, and input passages follow retrieval order. The 500 question IDs, dense retrieval outcomes, author-marked turns, model revisions and answer labels match the previous study.

| Evidence and question slice | Base calendar / offset / both | Large calendar / offset / both |
| --- | ---: | ---: |
| Dense, all 470 answerable | 25 / 14 / 5 | 51 / 49 / 44 |
| Dense, 127 temporal | 7 / 5 / 2 | 8 / 7 / 8 |
| Author-marked, 127 temporal | 12 / 14 / 14 | 14 / 11 / 14 |
| Dense, 92 temporal known by query date | 3 / 2 / 1 | 5 / 5 / 4 |
| Author-marked, 92 temporal known by query date | 9 / 10 / 10 | 9 / 9 / 10 |

Cells count exact answer and required annotated evidence jointly. On all answerable questions, adding the offset to the calendar label reduced dense base joint exact by 4.26 points [−6.17, −2.55]. For the large reader on the temporal dense slice, adding the offset gained one joint result and lost one [−2.36, 2.34 points]. Across 30 unknown questions, calendar and both markers each led the large dense reader to abstain on six. These scores describe lexical exactness. They leave open whether extracting and composing event dates from the conversation can help.

HPG L4 array 53000514. [Matched turn hashes and paired estimates](../../results/nonlinear_retrieval_natural/longmemeval_m_relative_clock_v1/) · [Earlier chronological ordering](LONGMEMEVAL_TIMELINE_ORDER.md)
