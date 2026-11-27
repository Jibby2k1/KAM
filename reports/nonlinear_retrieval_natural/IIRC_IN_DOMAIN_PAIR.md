# Learning one-pass pair retrieval on separate IIRC articles

With identical visible link pools, the IIRC-trained independent ranker recovered **56 of 153** gold pairs after adding source hyperlink context. A link-neutral pair head recovered **66**, and the pair head with between-candidate title features recovered **73**. The primary pair-versus-independent gain was **+6.54 percentage points** with a paired 95% source-article interval of **0.00 to +13.01**. [Fit and held-out replay](../../results/nonlinear_retrieval_natural/iirc_pair_fit_v1/)

| Frozen development slice | Questions | Frozen sequential | Learned independent, article leads | Learned independent, hyperlink context | Learned pair, link-neutral | Learned pair, title features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All two-link questions | 153 | 40 | 39 | 56 | 66 | 73 |
| Gold articles mention one another's title | 33 | 13 | 11 | 18 | 20 | 16 |
| No between-gold title mention | 120 | 27 | 28 | 38 | 46 | 57 |

The hyperlink-context gain within independent scoring was **+11.11 points** (**+3.87 to +18.67**, 25 wins and 8 losses). With identical hyperlink-context inputs, the neutral pair head added 16 wins and 6 losses over the independent head. Adding title features to that pair head produced **+4.58 points** overall (**−0.66 to +9.80**), with 11 gains in the no-gold-mention slice and four losses in the mention slice. These slices describe the gold evidence and use annotations. The frozen HotpotQA pair scorer previously achieved 22/153 on these same pools. The trained heads and frozen sequential baseline have different IIRC supervision. The matched independent-versus-pair comparison isolates the use of pair interactions among the fitted heads. [Frozen transfer](IIRC_PAIR_TRANSFER.md) · [Paired analysis](../../results/nonlinear_retrieval_natural/iirc_pair_fit_v1/summary.json)

## Fit and evaluation controls

The official train split provides 1,200 exact-two-linked-source questions from 1,055 original source articles. The 153-question evaluation uses the same sealed development cohort as the frozen study, from 117 original articles disjoint from the fitting set. The answerer's linked evidence titles define truth. The ordered initial candidate lists are the released hyperlinks that resolve in the official article archive. The evaluator checks each development pool fingerprint and gold slots against the earlier frozen replay. [IIRC paper](https://aclanthology.org/2020.emnlp-main.86/) · [Cohort](../../results/nonlinear_retrieval_natural/iirc_pair_v1/manifest.json)

The frozen E5 encoder and HotpotQA query adapter supply question, source-conditioned question, and linked article affinities. Hyperlink context uses the neighborhood of each released source anchor, limited to 20 words on each side of up to three mentions. The additive arms sum individual document features, which makes their highest pair equivalent to the top two document scores. The neutral pair head adds symmetric evidence features, article and hyperlink-context cosine, and lexical complement. The title arm also uses indicators for one-way and two-way title mentions in article leads. All arms use conditional softmax over every valid unordered pair and the same L2 penalty of 0.05 after training-only standardization. The fit was locked before reading development features. [Fit code](../../scripts/fit_kam_iirc_pairs.py) · [Locked coefficients](../../results/nonlinear_retrieval_natural/iirc_pair_fit_v1/fit.json)

HPG arrays **52955503** and **52955504** extracted train and development features on B200s, respectively. The four small ranking heads were fit once on CPU. Results measure pair selection given the released candidate sets. No reader or warmed serving latency was measured for these fitted arms. A matched, trained sequential baseline is needed to compare policy families under equal supervision. [GPU feature extractor](../../scripts/run_kam_iirc_pair_features.py) · [Slurm driver](../../slurm/nonlinear_retrieval_iirc_pair_features.sbatch)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_pair_fit_v1"
export OPENBLAS_NUM_THREADS=1
python scripts/fit_kam_iirc_pairs.py fit --train "$study/train-features.json.gz" \
  --output /tmp/kam-iirc-refit.json
python scripts/fit_kam_iirc_pairs.py evaluate --dev "$study/dev-features.json.gz" \
  --lock "$study/fit.json" --reference "$base/iirc_pair_v1/replay.json.gz" \
  --output /tmp/kam-iirc-fit-eval.json
diff -u "$study/summary.json" /tmp/kam-iirc-fit-eval.json
```

The archived features contain numerical inputs and row fingerprints. To regenerate them from the official release, use `prepare_kam_iirc_pair.py --with-link-context` separately with `--train` and `--dev`, then run the staged GPU extractor.
