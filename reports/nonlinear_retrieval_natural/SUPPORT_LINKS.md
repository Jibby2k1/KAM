# Question-conditioned pair scoring on held-out HotpotQA

The context-aware pair head ranked the annotated supporting link first on **316 of 360** questions that contained both a supporting link and linked distractors. The prior one-pass pair head did so on **209 of 360**. The new head also improved joint exact match over both prior retrieval baselines across the full split. The gain came from adding question-conditioned features. A separate loss restricted to linked pairs produced similar results.

## Held-out results

Each cell shows **both gold documents selected / joint exact match**. Joint exact match requires both gold documents and a normalized exact answer match. The linked and unlinked columns include only pools containing both gold documents.

| Selector | All 768 | Linked gold, 368 | Unlinked gold, 100 |
| --- | ---: | ---: | ---: |
| Learned single-shot top two | 117 / 74 | 94 / 64 | 23 / 10 |
| Learned sequential | 136 / 85 | 108 / 72 | 28 / 13 |
| Original one-pass pair scorer | 217 / 148 | 213 / 146 | 4 / 2 |
| Matched plain pair scorer | 211 / 144 | 209 / 142 | 2 / 2 |
| Context-aware pair scorer | **346 / 231** | **323 / 216** | **23 / 15** |
| Context-aware plus linked loss | 343 / **233** | 321 / **218** | 22 / **15** |

Both support documents entered the fixed top-64 dense pool for 468 of 768 questions. The other 300 contribute zero evidence and joint successes for every arm. The answer-only counts on those 300 were 15, 16, 16, 18, 29, and 31 in table order.

The context-aware scorer with linked loss improved full-split joint exact match by **11.1 percentage points** over the original pair head (paired 95% bootstrap interval **+8.7 to +13.5**) and **19.3 points** over sequential retrieval (**+16.3 to +22.4**). On unlinked gold, its evidence recovery was 6 points lower than sequential retrieval (**−18 to +6**) and joint exact match was 2 points higher (**−7 to +11**). The unlinked group is too small to resolve those differences.

## Supporting links versus linked distractors

Among 360 held-out questions containing a gold link and one or more linked distractors, there were 8,197 linked distractor pairs. This diagnostic restricts each pair head's ranking to linked candidates. Inference in the main table scores every candidate pair.

| Pair scorer | Gold link ranks first | Mean within-question win rate against wrong links |
| --- | ---: | ---: |
| Original | 209/360 (58.1%) | 88.7% |
| Matched plain | 201/360 (55.8%) | 89.3% |
| Context-aware | **316/360 (87.8%)** | **98.5%** |
| Context-aware plus linked loss | 314/360 (87.2%) | 98.4% |

The context-aware scorer with the linked loss gained 29.2 points over the original head on this diagnostic (**+24.2 to +34.4**), and differed from the same-feature scorer by −0.6 points (**−1.9 to +0.8**). In unrestricted inference on these 360 questions, the original head selected 151 wrong linked pairs, compared with 44 for the context-aware head and 46 with the added loss. Both context heads also improved support recovery over the original pair scorer on the unlinked stratum.

## Method

- Froze 1,024 fitting and 768 held-out bridge questions from the public HotpotQA training release before model fitting. The fixture excludes prior full-corpus cohort questions, the previous source questions used for router development, and reserved supporting titles. There are no overlaps in question ID, normalized question, or supporting title between fit and test. The fixture disallows held-out gold documents as training distractors. This is a new held-out split of public training data.
- Retrieved the top 64 from the same masked 5.2-million-document BEIR HotpotQA corpus using frozen E5-small-v2 and the FP32 dense index. The gold pair occurred naturally in 612 fitting pools. Added missing positives for fitting only. In 733 fitting questions, a gold link competed with at least one linked wrong pair. No positive was injected at evaluation.
- Used the same frozen query adapter, candidate embeddings, FLAN-T5-small checkpoint and reader token budget across arms. The sequential arm re-encoded the question with its first retrieved document and chose the second document from the same pool. Each pair arm exhaustively scored the 2,016 unordered candidates in one routing step.
- A link is an exact normalized title mention in the first 120 word tokens of either article. The matched plain pair head uses query affinities, document similarity, and title-link features. The two context-aware heads add distinct query-term coverage across the documents and overlap within 12 tokens of a title mention. They share architecture, initialization, fitting pools and minibatch order. The added objective gives equal weight to full-pool cross entropy and conditional linked-pair cross entropy on eligible fit rows.
- Reported paired 95% bootstrap intervals from 5,000 question-level resamples. These intervals measure variation across the test questions for one fit. The context feature bundle includes document coverage and mention-local overlap, so this experiment cannot isolate their individual contributions. It measures document-level evidence, with no human check that a matched mention expresses the relevant relation. Full-pipeline latency and transfer to other corpora remain unmeasured for this scorer.

## Reproduction

- [Recompute the report and reproduce the GPU study](../../docs/research/REPRODUCE_HOTPOTQA.md)
- [Fixed split](../../results/nonlinear_retrieval_natural/support_link_v1/fixture.json) · [Fitting audit](../../results/nonlinear_retrieval_natural/support_link_v1/fit-0.json) · [Machine-readable results](../../results/nonlinear_retrieval_natural/support_link_v1/summary.json)
- Held-out evaluations for [shard 0](../../results/nonlinear_retrieval_natural/support_link_v1/eval-0.json), [shard 1](../../results/nonlinear_retrieval_natural/support_link_v1/eval-1.json), [shard 2](../../results/nonlinear_retrieval_natural/support_link_v1/eval-2.json), and [shard 3](../../results/nonlinear_retrieval_natural/support_link_v1/eval-3.json)
- [HPG accounting](../../results/nonlinear_retrieval_natural/support_link_v1/slurm-accounting.psv) · [Fit source hashes](../../results/nonlinear_retrieval_natural/support_link_v1/fit-source.sha256) · [Evaluation source hashes](../../results/nonlinear_retrieval_natural/support_link_v1/eval-source.sha256)

Fitting used one HPG NVIDIA L4 and evaluation used four parallel L4 shards. Successful jobs consumed 227 GPU-seconds. Startup retries and an evaluation checkpoint metadata repair consumed another 122 GPU-seconds. The repair converted the PyTorch version object into a plain string and verified exact equality of every trained tensor before evaluation resumed. The reader loader kept distinct embedding and output weights from the pinned snapshot for every arm.
