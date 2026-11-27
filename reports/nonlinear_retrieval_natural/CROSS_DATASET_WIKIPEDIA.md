# Evidence coverage across the frozen Wikipedia index

The 5.20-million-document HotpotQA index contained every annotated article for 80.3% of the audited 2WikiMultiHopQA questions and 84.7% of MuSiQue-Ans. Verbatim evidence coverage was lower. The index held every annotated sentence for 19.8% of 2Wiki questions and every annotated paragraph for 6.2% of MuSiQue questions.

## Join and evidence checks

Audited all 12,576 records in the saved 2Wiki development split and all 2,417 in the saved MuSiQue-Ans development split. The join used the existing case-insensitive Unicode title key. For 2Wiki, each evidence unit uses the annotated sentence index in the dataset context. For MuSiQue, each marked supporting paragraph is an evidence unit. A normalized exact match requires the entire annotated unit to occur contiguously in the indexed text. An excerpt check accepts any contiguous 12-token window in the unit, or the whole unit when shorter. [2Wiki paper](https://aclanthology.org/2020.coling-main.580/) · [MuSiQue paper](https://aclanthology.org/2022.tacl-1.31/) · [BEIR HotpotQA corpus](https://huggingface.co/datasets/BeIR/hotpotqa)

| Cohort | Questions | All titles indexed | Excerpt from each unit | All units exact |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMultiHopQA | 12,576 | 10,096 | 4,772 | 2,488 |
| MuSiQue-Ans, all | 2,417 | 2,046 | 425 | 150 |
| MuSiQue-Ans, two hop | 1,252 | 1,116 | 338 | 138 |
| MuSiQue-Ans, three hop | 760 | 616 | 70 | 9 |
| MuSiQue-Ans, four hop | 405 | 314 | 17 | 3 |

The index's prior holdout exclusions removed at least one annotated title on 483 2Wiki and 177 MuSiQue questions. At least one required title was absent outside those exclusions on 2,052 and 203 questions, respectively. These causes can overlap within one question. Title presence measures availability at article resolution. Exact and excerpt checks measure retained annotation text. Sentence counts and paragraph counts describe different units, and paraphrased or alternative evidence can fall outside both checks.

The 2Wiki exact-evidence subset supports a follow-up on the existing index, alongside a title-complete cohort missing some annotation text. The small strict MuSiQue cohort reinforces the need for an independently assembled paragraph corpus for native long-hop evaluation. HPG CPU job 52967153 ran this audit. [Audit implementation](../../scripts/audit_kam_cross_dataset_wikipedia.py) · [Coverage counts and corpus fingerprints](../../results/nonlinear_retrieval_natural/cross_dataset_wikipedia_v1/summary.json)
