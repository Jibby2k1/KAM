# Passage windows improve frozen BRIGHT retrieval across four domains

With the first 188 body tokens as the only dense view, the hybrid ranked many relevant documents poorly even though those documents were already indexed. Embedding the tail and up to two interior windows raised nDCG@10 and recall@64 in economics, earth science, psychology and sustainable living. The same additional vectors assigned to length-matched wrong documents produced smaller gains or losses. Correct passage ownership mattered even with vector count and scoring held fixed.

| BRIGHT domain | Questions / documents | Hybrid nDCG@10, head → windows | Paired change, 95% interval | Hybrid recall@64, head → windows | Extra dense bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Economics | 103 / 50,220 | 15.79 → 19.05 | +3.26 pp [1.26, 5.56] | 32.70 → 35.62 | 24.3% |
| Earth science | 116 / 121,249 | 17.79 → 23.01 | +5.22 pp [3.06, 7.65] | 35.81 → 42.20 | 2.7% |
| Psychology | 101 / 52,835 | 10.61 → 15.01 | +4.40 pp [1.71, 7.44] | 20.81 → 27.95 | 19.6% |
| Sustainable living | 108 / 60,792 | 12.19 → 13.48 | +1.29 pp [0.20, 2.52] | 30.52 → 34.05 | 18.3% |

Psychology and sustainable living were reserved before scoring the first two domains. All conditions used frozen E5-base-v2 query embeddings, the same full-domain BM25, document exclusion rules, the same fusion, and exactly 64 output documents. Each indexed document retains its head embedding. Longer documents contribute up to three additional windows, each with the same E5 passage prefix and at most 188 content tokens. Dense rank uses the highest window score per document. Rank lists did not change when replaying the frozen baselines on all 428 questions. The paired bootstrap groups questions that share any gold document. The original BRIGHT revision, packed fixture hashes, window counts and device timing are recorded alongside the results. [Baseline and source revision](BRIGHT_FROZEN_TRANSFER.md) · [Preparation, runner and analyzer](../../scripts/run_kam_bright_passage_windows.py)

Three seeded controls reassigned extra windows among documents with the same number of extra windows and similar lengths. They kept the same head vectors, extra vector bank, per-owner window count and scoring procedure. Real windows beat all three controls separately in all four domains. The smallest lower endpoint of the 12 individual 95% intervals for nDCG@10 was +0.29 points. Tail only captured much of the effect in psychology (14.72 nDCG@10) and exceeded the four-window policy on sustainable living (13.80 vs 13.48 nDCG@10). More windows are therefore a coverage choice even at unchanged output depth.

At least one gold record exceeded the head cap on 92 economics queries, 114 earth science, 94 psychology and 88 sustainable living. The warmed L4 dense matrix with ordering took roughly 4.02 → 4.27 ms, 11.58 → 11.72 ms, 4.17 → 4.40 ms and 5.23 → 5.47 ms respectively in the control runs. These timers exclude source ingest, lexical retrieval, query embedding and reader work. This experiment measures evidence ranking. It does not measure joint answer accuracy. [Numerical outputs and replay audits](../../results/nonlinear_retrieval_natural/bright_passage_windows_v2/)

HPG L4 arrays 53002056/57, 53003099/100 and 53003538/39.
