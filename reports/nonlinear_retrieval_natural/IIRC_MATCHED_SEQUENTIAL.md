# Joint and sequential retrieval with matched IIRC supervision

On the same 153 IIRC questions and fixed candidate representations, the trained link-neutral joint head recovered **66** gold pairs and the trained first-then-second policy recovered **62**. The paired joint advantage was **2.61 percentage points**, with a 95% source-article interval of **−1.89 to +7.19**. With between-candidate title features, the corresponding counts were **73** and **65**. Both decoders used the same 1,200 IIRC training questions. [Matched replay](../../results/nonlinear_retrieval_natural/iirc_sequential_fit_v1/)

| Features and selection policy | All, n=153 | Gold articles mention one another, n=33 | No between-gold mention, n=120 |
| --- | ---: | ---: | ---: |
| Independent top-two with hyperlink context | 56 | 18 | 38 |
| Neutral joint | 66 | 20 | 46 |
| Neutral joint score restricted to fixed first | 63 | 18 | 45 |
| Neutral conditional fit, fixed first | 62 | 17 | 45 |
| Title-aware joint | 73 | 16 | 57 |
| Title-aware joint score restricted to fixed first | 63 | 14 | 49 |
| Title-aware conditional fit, fixed first | 65 | 15 | 50 |

The additive first hop picked a gold article on **91 of 153** questions. Given those correct first picks, the conditional heads recovered the other article on 62 and 65. Among cases with a wrong first pick, the unrestricted neutral joint scorer recovered six pairs, and the title-aware joint scorer recovered 12. Giving each conditional head the better of the two gold anchor choices would reach **110** and **116**, respectively. Those ceilings use annotations. They bound second-hop ranking when a correct first article is available. [Per-question outcomes](../../results/nonlinear_retrieval_natural/iirc_sequential_fit_v1/summary.json)

## Matched controls

Every arm consumes the same sealed official links and frozen E5-derived article and hyperlink representations from the prior B200 extraction. The independent ranker is the existing link-context additive model. Its best document supplies the first-hop choice to both sequential controls. The restricted control takes the frozen joint scores and searches only pairs containing that document, so the difference from unrestricted joint scoring is the first-hop commitment. The conditional model uses the same symmetric features but fits the second-hop distribution separately, training on each gold article as the first article. This gives 2,400 teacher-forced decisions across the same 1,200 questions. The per-choice loss and L2 penalty of 0.05 match the earlier fit, and the conditional heads were locked before evaluating their held-out decisions. They were fitted on CPU. [Joint fitting protocol](IIRC_IN_DOMAIN_PAIR.md) · [Conditional fit](../../results/nonlinear_retrieval_natural/iirc_sequential_fit_v1/fit.json)

For title-aware scores, restricting the unchanged joint head cost 10 cases (**−6.54 points**, paired 95% interval **−11.41 to −1.96**). Conditional refitting recovered two net cases relative to that restriction, leaving the conditional run eight behind unrestricted joint scoring (**−5.23 points**, **−10.42 to 0.00**). The neutral and title-aware evidence slices differ, and the comparison reuses the previously reported development cohort. It tests the cost of a hard first-hop commitment with fixed representations. A sequential encoder that builds a new question representation after seeing its first article is a separate experiment.

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
prior="$base/iirc_pair_fit_v1"
study="$base/iirc_sequential_fit_v1"
export OPENBLAS_NUM_THREADS=1
python scripts/fit_kam_iirc_sequential.py fit --train "$prior/train-features.json.gz" \
  --joint-fit "$prior/fit.json" --output /tmp/kam-iirc-conditional-refit.json
python scripts/fit_kam_iirc_sequential.py evaluate --dev "$prior/dev-features.json.gz" \
  --conditional-fit "$study/fit.json" --joint-fit "$prior/fit.json" \
  --reference "$prior/summary.json" --output /tmp/kam-iirc-conditional-eval.json
diff -u "$study/summary.json" /tmp/kam-iirc-conditional-eval.json
```
