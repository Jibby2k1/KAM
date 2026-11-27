# Native 2Wiki refit on a fixed shortlist

Fitting the same 16 pair features on native 2Wiki recovered both supporting articles on 425 of 1,024 held-out questions, up from 383 with the frozen HotpotQA scorer. Joint exact match with the same reader was 162 after the refit and 165 before. Most additional support recovery occurred where the original dense top 64 lacked a required article.

## Matched evaluation

The 512 fit questions shared zero required titles, IDs, or normalized question strings with the 1,024-question test fixture or the prior reserved questions. Each fit cohort contained 112 comparison, 128 compositional, and 16 inference cases. Both scorers used the same frozen conditioned 128-document pools, shortlist, pair features, and article ordering rule. The in-domain model changed only feature scaling and the 16 linear coefficients. A prespecified L2 penalty of 0.025 was used on 376 fit cases where the shortlist contained the positive pair. The fit process wrote the policy hash before the held-out predictions were scored.

| Held-out subgroup | Questions | Frozen support | Refit support | Frozen joint EM | Refit joint EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 1,024 | 383 | 425 | 165 | 162 |
| Original dense 64 complete | 590 | 311 | 313 | 145 | 138 |
| Original dense 64 incomplete | 434 | 72 | 112 | 20 | 24 |
| All annotated evidence present in the index | 512 | 222 | 239 | 106 | 103 |
| At least one annotated sentence absent | 512 | 161 | 186 | 59 | 59 |

Overall support increased by 4.10 percentage points (paired question-bootstrap 95% interval +1.95 to +6.35). Joint EM moved by −0.29 points (−1.46 to +0.78). On the original incomplete pools support increased by 9.22 points (+6.45 to +12.21) and joint EM by 0.92 points (+0.23 to +1.84), from four added successes and zero losses. The refit changed 325 ordered pairs and both models had the gold pair shortlisted on 711 questions.

## Reader control on ranking disagreements

| Gold pair recovered by | Questions | Joint EM under the scorer that recovered it | EM using fixed gold native leads | EM using annotated sentences |
| --- | ---: | ---: | ---: | ---: |
| Both | 339 | 147 | 149 | 169 |
| Frozen scorer only | 44 | 18 | 19 | 19 |
| 2Wiki refit only | 86 | 15 | 15 | 28 |

The rows join outputs from the independent [fixed-article reader control](NATIVE_TWOWIKI.md). Among the 86 added supports, the existing reader answered only 15 correctly even with both native gold articles. Feeding the same reader the annotated sentences raised that to 28. Evidence selection and reading remain separate sources of loss. The chance of joint success differed substantially between the two discordant groups.

The fit builder was HPG CPU job 52971444. Fit search 52971445, feature extraction 52971688 and 52971689, and fixed-reader comparison 52973063 ran on NVIDIA L4. The comparison reproduced all 3,072 archived reader exactness and F1 scores for the three prior policies, along with all 1,024 conditioned pair choices and shortlist flags. It then decoded 293 additional unique prompts. The archived features and reader results contain IDs and numeric data. The external corpus and question text are needed when rebuilding the frozen fixtures and executing the GPU jobs. [Locked policy, candidate features and paired replay](../../results/nonlinear_retrieval_natural/native_twowiki_refit_v1/) · [Fit and score runner](../../scripts/fit_kam_native_twowiki.py) · [Paired analysis](../../scripts/analyze_kam_native_twowiki_refit.py)
