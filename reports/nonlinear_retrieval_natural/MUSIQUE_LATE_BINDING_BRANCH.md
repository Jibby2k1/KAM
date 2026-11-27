# A second late binding at the same MuSiQue evidence budget

A confidence-accepted extractive alternative did not improve any four-hop joint answers over the original generative binding. Candidate coverage changed little. An unbound second search was worse than the reference at two and three hops on the same cases.

All three arms use the same author-supplied templates and generative reader until the penultimate hop. The primary path continues unchanged to the last subquestion. A lexical control queries an otherwise identical last subquestion with its penultimate reference removed. A second control fills that reference from the fine-tuned span reader on the same observed paragraph, using the established null threshold. It skips the extra search when the span reader abstains or gives the same text. Both alternatives use equal reciprocal-rank fusion with the primary last search. They perform an additional corpus scan on active requests. The total retained candidate limit remains 128 and only the last per-hop tranche can change. The final FLAN-T5-base reader sees the original question and exactly four distinct paragraphs, one from each available step plus any required slots backfilled by the candidate pool.

| Hop count | Questions | Primary, full pool / selected support / joint | Unbound branch | Span branch |
| --- | ---: | ---: | ---: | ---: |
| Two | 224 | 183 / 98 / 42 | 176 / 59 / 29 | 185–187 / 96–99 / 39–41 |
| Three | 192 | 120 / 30 / 15 | 120 / 13 / 1 | 123–124 / 25–29 / 11–15 |
| Four | 160 | 52 / 9 / 5 | 45 / 1 / 1 | 52 / 9 / 5 |

At four hops the span applied a distinct alternate search on 42–49 questions, depending on seed. All 160 final joint indicators matched the reference individually in all three fits. The unbound branch lost 13 joint matches on two hops and 14 on three. Paired shared-evidence cluster intervals were [−9.29, −2.62] and [−12.73, −2.89] percentage points. All 1,728 primary binding traces matched the archived baseline, and the alternatives preserved the entire pool before the final hop. This trial addresses late branching with a fixed reader. [The original early-binding intervention](MUSIQUE_SINGLE_BINDING.md) shows higher oracle headroom at the first binding for complete candidate recovery.

HPG L4 array 52998368. [Checksummed readouts and paired comparison](../../results/nonlinear_retrieval_natural/musique_binding_branch_v1/) · [Span fitting and abstention](MUSIQUE_SPAN_BINDINGS.md)
