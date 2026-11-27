# Frozen retrieval transfer to BRIGHT

Hybrid BM25 and E5 retrieval gave the highest nDCG@10 on both BRIGHT domains. The six MuSiQue-trained decomposers matched or slightly changed dense retrieval and trailed the hybrid on both domains. This measures transfer of a Wikipedia-trained decomposition policy to reasoning-intensive document search in independent economics and earth science corpora.

The external fixture contains all 103 economics queries with 50,220 documents and all 116 earth science queries with 121,249 documents from the pinned release. Every gold and excluded document resolves to the domain corpus. Only user questions and original corpus text go into the models. The ranking stage uses query-specific author exclusions, and the scorer uses gold document IDs.

Every arm outputs 64 unique documents. BM25 searches complete documents and E5 indexes their first 192 tokens. The two-window query control scores the query prefix and suffix through the same document bank. The conditioned arm mixes the question similarity with the similarity for a question supplemented by an excerpt from its first dense hit. The six frozen FLAN-T5 planners were fit on MuSiQue and validated there. They search by predicted subquestions, resolve reference variables with FLAN-T5-base against the top observed passage, and fuse those rankings with direct E5. Invalid plans use direct E5. This comparison holds output depth fixed. The policies have different numbers of scans and inference calls.

| Method | Economics nDCG@10 / recall@64 | Earth science nDCG@10 / recall@64 |
| --- | ---: | ---: |
| BM25 | 11.70 / 29.18 | 11.57 / 34.05 |
| E5 | 8.37 / 22.16 | 12.78 / 23.01 |
| E5 two-query-window maximum | 8.01 / 21.40 | 12.94 / 22.79 |
| One observed anchor | 8.61 / 23.62 | 12.09 / 24.06 |
| Hybrid BM25 and E5 | 15.79 / 32.70 | 17.79 / 35.81 |
| Frozen MuSiQue-small planner | 8.28–9.13 / 21.61–22.63 | 13.04–13.89 / 23.44–24.73 |
| Frozen MuSiQue-base planner | 7.89–8.50 / 22.29–22.72 | 11.26–12.22 / 23.29–24.01 |

Scores are percentages. Ranges are extrema across three frozen seeds. Hybrid nDCG@10 gains over BM25 were 4.09 percentage points [0.77, 7.44] on economics and 6.22 [2.50, 10.15] on earth science. Paired resampling keeps queries with shared gold documents together. Hybrid recovered every judged relevant document on 22 economics questions and 20 earth science questions. The corresponding ranges for learned plans were 14–15 and 15–16. In economics only 20–37 plans were syntactically valid out of 103, compared with 41–50 out of 116 in earth science. Validity checks reference indices and nonempty segments. A syntactically valid plan can still be irrelevant. BRIGHT supplies judged relevant documents, so these results concern retrieval rather than answer generation.

## Planning inside the same hybrid

A matched replay keeps the lexical rank weight at 1.0 and splits the direct dense rank weight between the original query and the additional ranks, with 0.5 total for each. The learned arm divides its 0.5 evenly across all valid subqueries. Invalid plans replay the ordinary hybrid. The same weighting rule applies to the anchor and window controls. All 219 earlier ranking records and plan traces reproduced when the new policies were evaluated.

| Hybrid with half the dense weight on an auxiliary search | Economics nDCG@10 / recall@64 | Earth science nDCG@10 / recall@64 |
| --- | ---: | ---: |
| Ordinary hybrid | 15.79 / 32.70 | 17.79 / 35.81 |
| Query windows | 16.15 / 32.04 | 17.68 / 35.98 |
| One anchor | 15.98 / 32.29 | 18.81 / 37.06 |
| Frozen MuSiQue-small plans | 15.32–15.86 / 31.10–33.02 | 18.12–19.10 / 35.37–37.30 |
| Frozen MuSiQue-base plans | 15.19–15.96 / 32.26–33.58 | 17.98–19.13 / 34.84–36.24 |

Some frozen plans improved early earth science rank by up to 1.34 points over the ordinary hybrid, with two of six seed-specific intervals above zero. The simple anchor gained 1.02 points [−0.22, 2.54], and recalled more gold documents overall than five of the six plan fits. The learned interventions recovered at most the same 20 fully covered questions there. All economics plan intervals span zero. The six fits are sensitivity analyses on one set of questions. These results support testing an inexpensive anchor against the learned augmentation at the same serving cost.

HPG L4 arrays 52996924, 52996925 and 52999962 and CPU array 52996923. [Ranks, checksummed manifests and paired summaries](../../results/nonlinear_retrieval_natural/bright_transfer_v1/) · [BRIGHT paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7a0f8055c838df8e62329a76c7c6403d-Paper-Conference.pdf) · [Pinned benchmark](https://huggingface.co/datasets/xlangai/BRIGHT/tree/8c3fa985016abc6b0e914d82dd01950e0669c0b7)
