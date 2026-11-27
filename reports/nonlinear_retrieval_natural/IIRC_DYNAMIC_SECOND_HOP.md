# Re-encoding the question for IIRC's second hop

Re-encoding the source-conditioned question after seeing the first article **did not improve exact pair recovery** on the 153 held-out IIRC questions. The neutral sequential score recovered 62 pairs with either static or re-encoded features. The title-aware version went from 65 to 63. This isolates second-hop features while keeping the first link fixed. [Paired audit](../../results/nonlinear_retrieval_natural/iirc_conditioned_fit_v1/summary.json)

| Features | Static sequential | Re-encoded second hop | Unrestricted joint |
| --- | ---: | ---: | ---: |
| Link-neutral, all 153 | 62 | 62 | 66 |
| Title-aware, all 153 | 65 | 63 | 73 |
| Link-neutral, 91 with a gold first | 62 | 62 | 60 |
| Title-aware, 91 with a gold first | 65 | 63 | 61 |

The neutral exact-recovery labels agreed on every question and the selected pairs agreed on 152. The title-aware re-encoding gained one case and lost three versus static features (**−1.31 percentage points**, 95% source-article interval **−3.97 to +1.28**). Its gap to the unrestricted title-aware pair head was **−6.54 points** (**−11.76 to −1.79**). With the better gold anchor supplied, the re-encoded heads could recover **109** neutral and **114** title-aware pairs. The analogous static ceilings were 110 and 116. [Static comparison](IIRC_MATCHED_SEQUENTIAL.md)

## What changed

The first-hop selector, candidate pools, article lead text, and preceding representations come from the prior matched IIRC experiments. For the second hop, the frozen E5 query encoder receives the question, source excerpt, and a 48-word excerpt containing the selected article title and lead. The learned HotpotQA adapter transforms that query. Its similarity to the prospective second article and to the prospective second source hyperlink excerpt supplies two new features. The conditional head fits those alongside the existing static features. [Extractor](../../scripts/run_kam_iirc_conditioned_features.py)

The two conditional heads had the same 1,200 fitting questions, teacher-forced gold anchors, per-anchor loss, and standardized L2 penalty of 0.05 as the static study. The 2,400 decisions supplied separate conditional labels for the two gold anchor orders. The fit was locked before collecting development scores. In evaluation, all operational policies used the unchanged predicted first article. Gold-anchor evaluations were scored separately as diagnostics. Arrays **52958317** and **52958318** extracted the added features on L4. The static features were archived from B200, and the fits ran on CPU. The study reuses the IIRC development cohort and measures fixed-pool evidence recovery. [Locked fit](../../results/nonlinear_retrieval_natural/iirc_conditioned_fit_v1/fit.json) · [HPG driver](../../slurm/nonlinear_retrieval_iirc_conditioned_features.sbatch)

Within the measured protocol, the extra query encoder pass offered no accuracy return. The 91/153 first-hop recall and the larger gold-anchor ceilings leave more room in selecting or hedging the first article. Any latency claim for that choice requires a different benchmark.

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_conditioned_fit_v1"
static="$base/iirc_pair_fit_v1"
reference="$base/iirc_sequential_fit_v1/summary.json"
export OPENBLAS_NUM_THREADS=1
python scripts/fit_kam_iirc_conditioned.py fit --static "$static/train-features.json.gz" \
  --conditioned "$study/train-features.json.gz" --output /tmp/kam-conditioned-fit.json
python scripts/fit_kam_iirc_conditioned.py evaluate --static "$static/dev-features.json.gz" \
  --conditioned "$study/dev-features.json.gz" --model "$study/fit.json" \
  --reference "$reference" --output /tmp/kam-conditioned-summary.json
diff -u "$study/summary.json" /tmp/kam-conditioned-summary.json
```
