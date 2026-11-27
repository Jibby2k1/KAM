# Extractive intermediate bindings on MuSiQue

Fine-tuning a small extractive reader substantially improved monitored span accuracy. Replacing every intermediate answer in a MuSiQue chain with its best span gave little improvement in the final result. Allowing it to defer to the generative reader recovered three or four additional four-hop joint results in each of three seeds, with low counts and intervals reaching zero.

The reader uses E5-small and a learned start/end head over a single retrieved paragraph. The comparison freezes or tunes E5-small, holding the initial head, fit and monitor rows, minibatch order, and 720 updates of 32 rows fixed within seed. Only the MuSiQue training split supplies fit and monitoring examples. No benchmark supporting title appears in either partition. The two training partitions also share no paragraph title or body. Of 17,058 fit and 1,104 monitor examples, 15,978 and 1,016 fit the locked input caps. Negatives are matched examples containing a paragraph without the answer span. The fixed zero margin compares the best admissible span with the null span. This threshold uses no test labels.

| Monitor measure | Frozen encoder | Fine-tuned encoder |
| --- | ---: | ---: |
| Forced span match, out of 508 positives | 80–84 | 321–325 |
| Span match with abstention, out of 508 positives | 5–6 | 293–301 |
| Correctly abstained, out of 508 negatives | 488–489 | 480–490 |
| L4 training time per seed, seconds | 13.4–13.6 | 46–47 |

The readout uses the 576-question native panel and author subquestion templates. Every arm gives the current reader the same first retrieved paragraph at the point of intervention. Once an intermediate answer changes, subsequent queries and evidence can change. The candidate budget is 128, with the first 64 from the original question unchanged. The last subanswer is always produced by FLAN-T5-base. Joint requires its exact match and complete supporting-paragraph coverage in the 128. The table gives complete coverage and joint on the same cases. Ranges summarize the three paired seeds and are descriptive.

| Hops | Questions | Generative reference | Fine-tuned span at first hop | Fine-tuned span at all intermediates | Fine-tuned span with generative fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two | 224 | 183 / 50 | 179–183 / 47–49 | Same as first | 181–183 / 48–50 |
| Three | 192 | 120 / 27 | 127–129 / 25–27 | 125–129 / 16–21 | 122–131 / 20–22 |
| Four | 160 | 52 / 7 | 51–53 / 6–7 | 39–42 / 6–7 | 52–59 / 10–11 |

Each encoder arm reproduced all 1,728 archived reference traces. On four-hop questions the tuned fallback improved first-binding exactness from 65 to 76–85, while penultimate-binding exactness ranged from 33 to 41 against a baseline of 37. The three four-hop joint deltas were +1.88, +2.50 and +2.50 percentage points. Shared-evidence cluster intervals were [0, 3.97], [0, 6.67], and [0, 6.90]. On three-hop questions the fallback lost five to seven net joint answers while increasing coverage. Candidate recovery and final answer utility can move in opposite directions after a binding change. A test with learned templates or a fixed final paragraph budget is still needed to interpret transfer.

HPG L4 arrays 52995612 and 52995613. [Fit, fixture and replay summaries](../../results/nonlinear_retrieval_natural/native_musique_binding_spans_v1/) · [Generative reader intervention](MUSIQUE_BINDING_READERS.md) · [Single-binding oracle](MUSIQUE_SINGLE_BINDING.md)
