# Distinct subquery assignments improve four-paragraph MuSiQue evidence

On the frozen 576-question MuSiQue panel, assigning a different paragraph to each predicted subquery raised joint exact match from 22–25 to 38–42 across three checkpoints. Most gains came from two-hop questions. Four-hop joint exact match was zero throughout.

Each checkpoint produced the same 128-entry candidate list, intermediate predictions, and subqueries as in the previous replay. The rerun recorded each subquery's cosine affinity to every member of that list and verified all previous numerical records after excluding the new affinity arrays. Scores were standardized within each subquery. The joint policy maximized summed affinity subject to selecting one distinct paragraph per subquery. When there were fewer than four subqueries, it filled the remaining slots by highest standardized affinity. Invalid plans kept the hybrid fallback. Assignment uses the generated subqueries, predicted bindings, and candidate scores only.

| Checkpoint | Direct support / joint EM | Joint assignment support / joint EM | Two-hop joint EM, direct → assigned | Four-hop support / joint EM, assigned |
| --- | ---: | ---: | ---: | ---: |
| 106331 | 61 / 22 | 84 / 42 | 18 → 35 | 1 / 0 |
| 106339 | 64 / 22 | 87 / 38 | 20 → 32 | 0 / 0 |
| 106367 | 68 / 25 | 86 / 38 | 21 → 32 | 0 / 0 |

Matched changes in overall joint EM were +3.47, +2.78 and +2.26 percentage points. Shared-title cluster intervals were [+1.68, +5.62], [+1.14, +4.71] and [+0.80, +3.97]. The comparator used the same four-paragraph budget, original-question FLAN-T5-base reader, paragraph cap and aliases. Every archived reader control reproduced its evidence and exactness record. Hybrid reached 18 joint exact matches on the same panel, and selecting gold evidence available inside the learned pools reached 93–94.

Greedy distinct assignment in subquery order gave the same complete-support totals as joint optimization. Joint assignment changed the chosen step assignments on 53–65 questions and retrieved 1–2 extra gold paragraphs per seed in aggregate. The independent maximum-affinity policy recovered 78–83 full chains, compared with 84–87 for either distinct assignment. Joint assignment was selected in advance for the reader comparison. Greedy reader outcomes are not measured here.

These are paired results conditional on a previously examined evaluation panel. A separately sealed title-disjoint replication is needed for the two-hop result. The four-hop gap remains concentrated upstream as well as at final selection.

[Arms and numerical evidence](../../results/nonlinear_retrieval_natural/native_musique_affinity_v1/) · [Assignment audit](../../scripts/analyze_kam_musique_assignment.py) · [Previous fitted selector](MUSIQUE_EVIDENCE_SELECTOR.md)
