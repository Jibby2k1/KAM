# Persistent support banks under typed revision routing

Adding the KAM learned support bank to the guarded rank decoder did not improve strict archive accuracy consistently. The radial bank outperformed a similarly sized pointwise residual on both Eurostat indicators, and its paired intervals versus context-only included zero. Disabling the learned bank after training left most outcomes unchanged. These observations concern static learned supports in a task that supplies explicit temporal records and address rank.

The three variants start with identical common weights per geometry. The context-only decoder matches the archived hard-clock and rank fit. The support-bank decoder adds 24 learned keys and values per head, with soft routing in training. The pointwise control adds a bias-free 64→88→64 GELU residual in the same layer location and keeps the same surrounding normalization. Dot support and pointwise each have 117,056 parameters. The radial versions differ by 136 parameters out of about 117,000. All models receive 3,200 updates and 10,556,800 tokens at the same schedule and record order. All context-only sampled losses and per-example outputs replayed.

| Geometry and residual | Training time on L4 | Depth-1,024 synthetic corrections / 1,536 | Dallas depth 48 / 3,504 | Eurostat unemployment / 3,456 | Eurostat industry / 3,456 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dot, context only | 15.51 s | 1,474 | 2,125 | 2,766 | 3,152 |
| Dot, support bank | 20.82 s | 1,491 | 2,102 | 2,742 | 3,155 |
| Dot, pointwise | 17.26 s | 1,507 | 2,120 | 2,777 | 3,160 |
| Radial, context only | 21.00 s | 1,436 | 2,043 | 2,850 | 3,151 |
| Radial, support bank | 31.34 s | 1,440 | 2,001 | 2,814 | 3,131 |
| Radial, pointwise | 22.94 s | 1,444 | 2,026 | 2,682 | 3,077 |

Radial support-bank gains over the pointwise comparator were +3.82 percentage points [0.95, 7.64] in unemployment and +1.56 [0.61, 2.78] in industry. Both pointwise results were below radial context-only. Compared with context-only, the support-bank interval spans zero on Dallas and the two original Eurostat panels for both geometries. The archive counts include three fits and two recodings of each query.

Frozen inference with the bank residual zeroed changed the deep correction count by −1 dot and −2 radial. Keeping the top four support keys per head made the same aggregate changes, while top one lost 10 dot and 42 radial joint outcomes. Dot's Dallas top-one loss was −1.11 points [−1.84, −0.40]. These frozen controls describe the reliance of the fitted models on their support reads. They do not evaluate models trained with a hard-routing objective.

HPG L4 array 53016424. [Paired data and intervals](../../results/nonlinear_retrieval_natural/guarded_support_bank_v1/) · [Context-only replay and rank setup](TYPED_CLOCK_GUARD.md)
