# Fresh two-hop MuSiQue replication of distinct evidence assignment

On 140 new two-hop MuSiQue questions, selecting one distinct paragraph per generated subquery increased joint exact match from 12–14 to 21–24 across three checkpoints. With the same four-paragraph reader budget, hybrid ranking achieved six joint matches. Greedy and global distinct assignment produced the same per-question joint decisions across all three seeds.

The evaluation contains every eligible question after removing repeated support sets and normalized question strings. The required titles are disjoint from the frozen 576-question assignment study, decomposer fitting, and its monitor, which also supplied the earlier selector fit. Six previous query fixtures were additionally reserved by question ID and normalized wording. The corpus, checkpoints, passage budget, original-question reader, and exact statistic were reused. The primary two-hop set has 140 questions. Seven additional three-hop questions are descriptive. The reservations left no eligible four-hop question.

| MuSiQue arm | Full support | Joint EM |
| --- | ---: | ---: |
| Hybrid top four | 24 | 6 |
| Original learned paths, three seeds | 39 / 41 / 38 | 13 / 14 / 12 |
| Greedy distinct assignment | 57 / 59 / 52 | 23 / 24 / 21 |
| Global distinct assignment | 58 / 60 / 53 | 23 / 24 / 21 |
| Gold selection within learned pools | 103 / 104 / 100 | 33 / 33 / 30 |
| Gold-only reference | 140 | 59 |

Per-seed joint improvements for global over the original learned path were +7.14, +7.14, and +6.43 percentage points. Shared-evidence cluster intervals were [+3.33, +11.54], [+1.45, +13.29], and [0.00, +13.67]. Global assignment recovered one additional complete evidence set in each seed relative to greedy assignment. Those additions produced no newly exact answers. This replicates a two-hop selection effect and supports using the simple greedy assignment as a deployment comparison. The high-hop question remains open.

HPG arrays 52984644, 52985678, and 52985679 ran the baseline, frozen subquestion generation, and matched scoring. [Numerical evidence and frozen panel](../../results/nonlinear_retrieval_natural/native_musique_assignment_replication_v1/) · [Primary assignment experiment](MUSIQUE_SUBQUERY_ASSIGNMENT.md)
