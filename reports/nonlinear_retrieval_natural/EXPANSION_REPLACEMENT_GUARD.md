# Guarding pair replacements after conditioned corpus expansion

## Design

The [conditioned union study](CONDITIONED_UNION.md) found a net gain from corpus expansion and a regression on complete pools. We test whether an inference-visible rule can retain broad's answer when the expanded reranker is uncertain about replacing its evidence. We kept the retrieval index, query adapter, reranker and reader frozen.

Broad ranks the original 64 articles and conditioned union ranks the original pool plus unique articles from the conditioned corpus top 64. The guard keeps broad when the union's pair contains only original articles. For other pairs it scores 14 features, including number and corpus rank of new documents, pair overlap, anchor membership, shortlist and title-link signals, reranker margins, and shifts in sentence relevance. No support labels or answer outputs enter its decision. The reader is shared by all arms. Joint exact match requires a correct pair and an exact normalized answer.

Two L2 logistic models use the 2,048 hash-selected development questions from the archived sequential fallback cohort. Each model learns the relative preference on cases with opposite support or opposite joint outcomes. The guard averages the two logits. The remaining 1,024 development questions calibrate a single threshold from five eligibility budgets. Calibration requires at least two additional joint successes overall and permits at most a 0.5 percentage point loss on each complete-pool metric. Viable thresholds are ordered by joint gain, then support gain, then fewer switches. Original-pool completeness enters calibration and evaluation only. A separate 1,024-question split, disjoint in IDs, normalized questions, and gold titles from 20 reserved fixtures, supplies the held-out evaluation.

The calibration chose the 50% budget. It retained 79 of 159 eligible replacements, gained 33 support recoveries and 23 joint exact matches overall, and lost one support and zero joint matches on the 632 complete pools. The absolute threshold was locked before the held-out reranking job.

| Eligible calibration cases admitted | Switched | Overall support / joint EM delta | Complete support / joint EM delta | Qualified |
| ---: | ---: | ---: | ---: | --- |
| 25% | 39 | +23 / +16 | −1 / 0 | Yes |
| 50% | 79 | +33 / +23 | −1 / 0 | Yes |
| 75% | 119 | +34 / +25 | −5 / −1 | No |
| 90% | 143 | +35 / +27 | −8 / −2 | No |
| 100% | 159 | +35 / +26 | −10 / −4 | No |

The guard acts after both scorers. This experiment measures accuracy and switch rates. It does not measure a latency reduction.

## Held-out results

| Original pool | Broad | Conditioned union | Added article required | Learned guard |
| --- | ---: | ---: | ---: | ---: |
| Complete, 606 | **464 / 297** | 456 / 295 | 455 / 294 | 461 / 296 |
| Incomplete, 418 | 0 / 0 | **56 / 46** | **56 / 46** | 39 / 29 |
| All 1,024 | 464 / 297 | **512 / 341** | 511 / 340 | 500 / 325 |

Cells count both support articles recovered / joint exact match. Against broad, the guard gained **3.52 points** in support (paired 95% interval **+2.34 to +4.69**) and **2.73 points** in joint EM (**+1.76 to +3.81**). Against always taking the conditioned union, it lost **1.17 support points** (**−2.25 to −0.20**) and **1.56 joint EM points** (**−2.54 to −0.68**).

The guard accepted 74 of 177 eligible replacements. On complete pools it finished within three support recoveries and one joint match of broad and gained five and one over the union. On incomplete pools it rejected 17 valid support replacements and 17 joint exact matches. The two-arm oracle reached 524 support and 346 joint exact matches, only 12 and five above the union. The unconditional union remains the higher-quality overall reference. The guard provides a measured option when preservation on initially complete pools takes priority.

## Reproduce

HPG arrays **52948760**, **52948773**, and **52949304** scored development, froze the held-out pools, and scored the held-out policies. The development run reproduced all 3,072 archived broad pairs and reader labels. The [frozen policy, provenance and replays](../../results/nonlinear_retrieval_natural/expansion_guard_v1/) and [Slurm driver](../../slurm/nonlinear_retrieval_expansion_guard.sbatch) record the study.

```bash
base=results/nonlinear_retrieval_natural
study="$base/expansion_guard_v1"
train="$base/sequential_fallback_v1"
fitted="$base/native_sentence_refit_v1/fitted.json"
options=(--fixture "$train/fixture.json.gz" --pools "$train/pools.json.gz" --fitted "$fitted")
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_expansion_guard.py lock "$study/training-replay.json.gz" "${options[@]}" --output /tmp/kam-expansion-policy.json
diff -u "$study/policy.json" /tmp/kam-expansion-policy.json
options=(--fixture "$study/fixture.json.gz" --pools "$study/pools.json.gz" --fitted "$fitted")
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_expansion_guard.py evaluate "$study/replay.json.gz" "${options[@]}" --policy "$study/policy.json" --output /tmp/kam-expansion-guard.json
diff -u "$study/summary.json" /tmp/kam-expansion-guard.json
```
