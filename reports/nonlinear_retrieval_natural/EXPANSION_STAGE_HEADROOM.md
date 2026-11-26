# Where the HotpotQA expansion pipeline loses evidence

The best measured arm in the matched B200 study, hybrid search with broad-ranked expansion, reached **340 joint exact matches** on 1,024 questions. An oracle choosing the best output of the four existing policies per question would reach **361**. Pooling both expanded candidate lists before ranking would make both gold articles available on **797** questions, versus **780** for hybrid expansion alone. This audit reuses the matched cohort to separate the opportunities for better search, ranking, and reading. [Matched B200 study](DENSE_HYBRID_ANCHOR.md) · [Numeric audit](../../results/nonlinear_retrieval_natural/expansion_stage_headroom_v1/summary.json)

## First failure in the hybrid expansion pipeline

| Original hybrid pool | Questions | Both gold unavailable after expansion | Gold pair absent from shortlist despite coverage | Wrong pair selected despite shortlist | Answer inexact despite correct pair | Joint EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Both supports present | 659 | 0 | 62 | 123 | 178 | 296 |
| One support missing | 353 | 232 | 23 | 29 | 25 | 44 |
| Both supports missing | 12 | 12 | 0 | 0 | 0 | 0 |
| All | **1,024** | **244** | **85** | **152** | **203** | **340** |

Each question is assigned to the first stage that fails. The groups use the original hybrid pool, before adding conditioned hits. Identifying these stages in hindsight does not supply a deployable completeness signal. On initially complete pools, **185** questions lost the gold pair during ranking and **178** had a reader error after a correct pair. On originally incomplete pools, **244 of 365** still lacked at least one article after expansion. [Replay summary](../../results/nonlinear_retrieval_natural/expansion_stage_headroom_v1/summary.json)

## What the existing policies could recover

| Policy or diagnostic oracle | Both gold in candidates | Gold pair shortlisted | Correct pair | Joint EM |
| --- | ---: | ---: | ---: | ---: |
| Hybrid with broad-ranked expansion | 780 | 695 | 543 | 340 |
| Choose between dense and hybrid expansions | 794 | 716 | 564 | 353 |
| Choose among both broad and both expanded policies | 794 | 727 | 579 | 361 |
| Merge dense and hybrid expanded candidates | 797 | — | — | — |

The choice-oracle figures are separate ceilings for each metric based on observed decisions. The joint oracle adds **14** successes in initially complete pools and **7** where one support was missing. The candidate merge increases the mean pool to **141.46** articles and covers three questions on which neither expansion separately contained both supports. It still completes none of the 12 pools originally missing both. The merged lists were evaluated only for article coverage. [Deterministic audit](../../results/nonlinear_retrieval_natural/expansion_stage_headroom_v1/summary.json)

The audit points to search changes for the still-incomplete cases and to ranking and reader changes when both documents are present. A selector among existing policies can address at most the 21 observed joint misses that another policy solved. Evaluate a new search or reranker on a separate held-out cohort before fitting another gate to these four policies.

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_expansion_stage_headroom.py \
  "$base/dense_hybrid_anchor_v1" --output /tmp/kam-expansion-stage-headroom.json
diff -u "$base/expansion_stage_headroom_v1/summary.json" /tmp/kam-expansion-stage-headroom.json
```
