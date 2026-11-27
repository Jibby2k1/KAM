# A small first-link beam on source-held-out IIRC questions

On 1,200 IIRC questions scored out of fold, using four possible first links raised the neutral sequential policy from **539 to 563** recovered gold pairs. Title-aware recovery rose from **547 to 598**. The four-link policies were within paired intervals spanning zero of their respective exhaustive joint heads. The primary two-link comparison for the neutral model gave **538 versus 539** for one link. [Cross-fitted results](../../results/nonlinear_retrieval_natural/iirc_anchor_beam_v1/summary.json)

| First-link budget | At least one gold first link | Neutral both gold | Title-aware both gold | Mean distinct unordered pairs within budget |
| --- | ---: | ---: | ---: | ---: |
| One | 777 | 539 | 547 | 13.30 |
| Two | 983 | 538 | 564 | 25.61 |
| Four | 1,116 | 563 | 598 | 47.22 |
| Exhaustive joint | — | 548 | 602 | 101.18 |

At width four, the paired gains over one link were **+2.00 percentage points** for neutral (95% source-article interval **+0.59 to +3.45**) and **+4.25 points** for title-aware (**+2.86 to +5.74**). The differences from their exhaustive joint heads were **+1.25** (**−0.08 to +2.60**) and **−0.33** (**−1.82 to +1.14**) points. For the primary neutral two-link test, the paired change was **−0.08 points** (**−1.57 to +1.41**). The neutral four-link branch oracle could recover **801** pairs by selecting the best conditional top-one branch after seeing the gold labels. The title-aware branch oracle reached **844**. [Paired counts, fold summaries and decisions](../../results/nonlinear_retrieval_natural/iirc_anchor_beam_v1/)

## Fixed folds and scoring

The official train examples were split by original source-article ID using a fixed SHA-256 rule. Each example was scored only by models fitted on the other four folds. This yielded five held-out groups of 215–253 questions, totaling 1,055 unique source articles. The development questions used in the previous reports remained outside this diagnostic. The numerical feature archive comes from the earlier B200 extraction. [Feature provenance](IIRC_IN_DOMAIN_PAIR.md)

The additive first scorer, exhaustive joint heads, and teacher-forced conditional second heads were refit per fold using the previous feature definitions and the same L2 penalty. Each retained first document contributes its normalized first probability times a normalized conditional probability over prospective partners. If both orders reach the same unordered pair, their probabilities are added. Selection breaks ties by stable pair index. [Fixed-feature training](IIRC_MATCHED_SEQUENTIAL.md) · [Cross-fit script](../../scripts/fit_kam_iirc_anchor_beam.py)

The four-link candidate subset contains **46.7%** as many distinct pairs as the exhaustive universe on average. This is a work-count proxy. The numerical replay computed dense feature matrices and never benchmarked a sparse serving implementation. Runtime savings require scoring only the retained branches in a separate benchmark. These are source-disjoint folds within an existing archive, and the paired intervals summarize the fixed out-of-fold predictions.

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_anchor_beam_v1"
OPENBLAS_NUM_THREADS=1 python scripts/fit_kam_iirc_anchor_beam.py \
  --static "$base/iirc_pair_fit_v1/train-features.json.gz" \
  --output /tmp/kam-iirc-beam-summary.json --replay /tmp/kam-iirc-beam-replay.json.gz
diff -u "$study/summary.json" /tmp/kam-iirc-beam-summary.json
```
