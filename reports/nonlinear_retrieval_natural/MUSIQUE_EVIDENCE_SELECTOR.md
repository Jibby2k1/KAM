# A title-disjoint learned selector for four final paragraphs

A separately fitted listwise selector raised MuSiQue joint exact match from 22–25 to 26–27 of 576 across three frozen decomposers. All matched overall intervals spanned zero. Most gains came from two-hop questions. No policy recovered the complete final four-hop evidence chain.

The fit split uses 208 questions from the decomposer monitor. Their required titles are disjoint from the decomposer fit and the frozen 576-question evaluation. HPG L4 generated the same 128-entry learned pools for fitting; the selector fit used 192, 185 and 195 questions with valid plans and at least one positive. Each seed-specific selector fits a regularized listwise objective over 12 inference-only features with a fixed penalty. Test selection returns four paragraphs from the same 128-entry pool, uses the hybrid fallback for invalid plans, and shares the previous FLAN-T5-base reader, prompt and alias matching. All archived control reader records matched exactly in the replay.

| Policy | Complete support | Joint EM | Two-hop joint EM | Three-hop joint EM | Four-hop joint EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid rank | 44 | 18 | 18 | 0 | 0 |
| Learned path, seed 106331 | 61 | 22 | 18 | 4 | 0 |
| Learned selector, seed 106331 | 57 | 26 | 25 | 1 | 0 |
| Learned path, seed 106339 | 64 | 22 | 20 | 2 | 0 |
| Learned selector, seed 106339 | 64 | 27 | 26 | 1 | 0 |
| Learned path, seed 106367 | 68 | 25 | 21 | 4 | 0 |
| Learned selector, seed 106367 | 64 | 27 | 26 | 1 | 0 |

Differences versus the corresponding learned path were +0.69, +0.87 and +0.35 percentage points in joint EM, with shared-title cluster intervals [−0.72, +2.23], [−0.37, +2.29] and [−0.99, +1.80]. Partial gains at two hops coexisted with lower three-hop joint counts for all seeds. The within-pool gold control recovered 261–266 full chains and 93–94 joint matches, which leaves a substantial selection gap at this context budget. A useful next diagnostic is to compare independent ranking with distinct per-step assignment on the same branch affinities.

[Frozen reader comparison](MUSIQUE_FINAL_READER.md) · [Fit and scoring](../../scripts/fit_kam_musique_selector.py) · [Numerical evidence](../../results/nonlinear_retrieval_natural/native_musique_selector_v1/)
