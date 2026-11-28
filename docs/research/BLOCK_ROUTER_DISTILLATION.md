# Can a small routing fit recover more dense attention?

Small per-head score corrections kept 0.13 percentage points more of the dense attention mass on held-out PG-19 development books. The exact fixed-budget oracle added 0.75 points over the current shortlist. The limited aggregate change and the distinct distant-evidence failure favor a targeted evidence diagnostic before adding a new kernel path.

## Fixed comparison

- Partition the official PG-19 validation books by hash into separate fit and development groups after token-length eligibility. Freeze two nonoverlapping 4K windows per book where available. Reuse the three frozen Llama checkpoints and keep full Flash active while collecting the reference queries and keys. [PG-19](https://github.com/google-deepmind/pg19)
- Measure the first query of a 32-token routing tile every 128 tokens, starting at token index 512. For each eligible older block, compute the exact dense log sum of exponentiated query–key scores. Leave block zero, the current block and the preceding block fixed. Allocate the remaining five slots by the compared ranking.
- Fit a weighted centered least-squares correction to the log mass in each layer and head. Compare maximum and mean of the four proxy scores, then the same measures with inverse-distance and inverse-block-index features. Use the same key summaries and a fixed small ridge toward the existing score.
- Compare the original protected kernel's actual shortlist, the two fitted rankings, a min/max upper-bound ranking, and the exact block-mass oracle. The min/max score adapts the page-ranking principle from Quest to this prefill diagnostic. The oracle ranks blocks with direct access to the dense scores. [Quest](https://arxiv.org/abs/2406.10774)
- Use disjoint books to compare retained mass across fit variants. A coverage gain is a prerequisite for integrating the correction and testing end-to-end quality and latency. Static head-level patterns and dynamic indices have been studied in long-context prefill, so position-aware gains here need matched downstream validation. [MInference](https://arxiv.org/abs/2407.02490)

## Results

The fit uses 37 books and the held development comparison uses 13 separate books from the official PG-19 validation split. Each book contributes up to two nonoverlapping 4K windows. Every row uses the same three checkpoint seeds and 28 sampled queries per window. Coverage averages books, seeds, sampled queries, layers and heads with equal weight at each averaging step.

| Ranking | Mean dense mass retained | Change vs. protected, percentage points |
| --- | ---: | ---: |
| Three mandatory blocks alone | 81.183% | −10.253 [−10.614, −9.922] |
| Min/max upper bound | 90.503% | −0.934 [−1.097, −0.764] |
| Protected proxy | 91.436% | Reference |
| Fitted maximum and mean | 91.516% | +0.080 [+0.072, +0.087] |
| Fitted moments and positions | 91.570% | +0.133 [+0.086, +0.199] |
| Exact block-mass oracle | 92.190% | +0.754 [+0.645, +0.894] |

Paired 95% intervals resample held-out books conditional on the three checkpoint seeds. The oracle is exact for this cost, fixed mandatory blocks and sampled dense query–key states. Its mean advantage reaches 3.45 points in layer 2 and 1.91 in layer 1. It gives a coverage ceiling for the sampled dense forward pass. Sparse model activations, end-to-end perplexity, and retrieval require separate experiments.

[Per-layer and per-head results](../../results/general_benchmarks_route_distillation/summary.json) · [Completed HPG fits](../../results/general_benchmarks_route_distillation/scheduler.tsv)
