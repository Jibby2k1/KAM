# Temporal multiple-choice recovery on TIME-Lite

A frozen reader did better with retrieved text than with no context on the news and dialogue panels. At identical selected text, chronological order improved the base dialogue result and the large news result. Showing versus masking the supplied dates did not give a consistent gain. The larger reader consumed about 3.3 times the reader time and answered fewer news questions correctly from the same dense evidence.

| Scenario and reader | Questions | No context | Prefix windows | Dense rank order | Same windows in chronological order | Same windows with dates masked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dialogue, base | 220 | 63 | 78 | 85 | 92 | 81 |
| Dialogue, large | 220 | 71 | 73 | 89 | 89 | 92 |
| News, base | 210 | 84 | 133 | 140 | 144 | 142 |
| News, large | 210 | 95 | 119 | 124 | 130 | 125 |
| Wiki, base | 66 | 23 | 23 | 26 | 24 in source order | — |
| Wiki, large | 66 | 26 | 29 | 26 | 26 in source order | — |

The frozen public release contains 1,549 context variants. The evaluation keeps one source setting per scenario and all 496 questions with an authored choice label and three or four valid options. News uses the authors' hybrid context, dialogue and wiki their base context. Wiki lacks separately extractable record date headers, so its order comparison reconstructs source order. The release file contains 330 base dialogue rows, while the paper's overview lists 322. The reported accuracy denominators follow the pinned release and explicit multiple-choice selection. [TIME benchmark paper](https://arxiv.org/abs/2505.12891) · [Pinned TIME-Lite source](https://huggingface.co/datasets/SylvainWei/TIME-Lite/tree/cd93b48ffeb8491f6d46359a8e2bf79495cb0ad3)

Each candidate window contains up to 140 words at stride 112 inside a source segment. A frozen E5 query embedding selects up to four windows from the supplied context. Both FLAN-T5 sizes get identical 1,024-token-bounded prompts. The date comparisons retain the same selected body tokens for each question and option order. Date omission masks fixed header slots. A distinct control cyclically rotates the date labels across the same windows, which changes at least one date header on 219 dialogue questions and 152 news questions. It changes these accuracy counts by at most three. Publication dates, session dates and dates mentioned inside the text are different information. The intervention touches the first two and leaves textual dates visible.

Scores average first-token option logits over every cyclic rotation of the option order and take the resulting best option. There are 30 dialogue, 53 news and 30 wiki clusters after joining questions that share source segments. For base dialogue, chronological order added 7 correct, a +3.18 point change with a shared-source paired 95% interval of [0.46, 5.88]. For large news, ordering added 6, +2.86 [0.43, 5.78]. On news dense-rank prompts, large versus base changed accuracy by −7.62 points [−14.64, −0.93], and used 137.57 versus 42.04 seconds for all rotations. These are multiple-choice decisions on the supplied evidence settings. They do not measure open retrieval support or recall.

HPG L4 array 53007184. [Source, exact prompt checks and paired numerical outcomes](../../results/nonlinear_retrieval_natural/time_lite_context_v1/) · [Previous long-history test](LONGMEMEVAL_TIMELINE_ORDER.md)
