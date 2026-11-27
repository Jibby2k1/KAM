# Four-paragraph readout after decomposer scaling

A larger learned decomposer raised complete candidate recovery on four-hop MuSiQue. With the final reader limited to four paragraphs, joint exact outcome counts stayed between zero and two. On the 576-question panel overall, greedy subquery assignment with the larger decomposer raised joint matches to 44–61, compared with 37–45 for the smaller decomposer.

The comparison uses the same three pairs of fitted FLAN-T5-small and base decomposers as the [candidate coverage study](MUSIQUE_DECOMPOSER_SCALE.md). The intermediate reader, paragraph encoder, candidate budget of 128, and final FLAN-T5-base reader are fixed. Both sizes use the same greedy distinct paragraph assignment and the same four-slot budget. The original question is the final reader query. The 576 native questions reproduced all 3,456 prior candidate rank records. The 140 primary fresh questions share neither evaluation questions nor supporting titles with the original set, and their support titles were excluded from fit and monitoring.

| Cohort | Small complete pool | Base complete pool | Small greedy support / joint | Base greedy support / joint |
| --- | ---: | ---: | ---: | ---: |
| Native, all 576 | 264–266 | 284–289 | 74–89 / 37–45 | 95–114 / 44–61 |
| Native, four hops, 160 | 16–21 | 25–29 | 1–2 / 1–2 | 2–4 / 0–2 |
| Fresh, two hops, 140 | 101–105 | 103–105 | 50–54 / 19–21 | 56–61 / 21–24 |

On all 576, paired gains for greedy base over greedy small were 16, three and seven joint answers. The first seed's interval excluded zero. All three intervals spanned zero on the fresh primary cohort, where the base gains were three, one and three. Assignment still improved joint matches over the same base decomposer's direct path in all three seeds on both cohorts. Replacing the four selected paragraphs with a budget-matched oracle from the same base candidate pool yielded 11–14 four-hop joint matches. More training or stronger decomposition can improve candidate coverage, while use of the available paragraphs remains a separate problem.

HPG arrays 52990987, 52990988 and 52990989. [Frozen affinity and readout records](../../results/nonlinear_retrieval_natural/native_musique_scale_readout_v1/) · [Earlier independent assignment](MUSIQUE_ASSIGNMENT_REPLICATION.md) · [Intermediate reader effects](MUSIQUE_BINDING_READERS.md)
