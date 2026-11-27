# Terminal evidence dependence in four-hop MuSiQue

On 160 four-hop questions, FLAN-T5-large answered 98 from the final gold passage body alone, 91 from all four gold passages, one from the three earlier gold passages, and zero from the question alone. A literal accepted answer appeared within that terminal body window in 153 of the 160 examples. The author supplies the terminal role. The comparison measures what the frozen reader uses after receiving known evidence.

The original question, versioned model checkpoints, annotation IDs and 192-token passage budgets match the preceding reader-window study. The full-passage and final-passage token hashes and all 640 base/large exact outcomes replayed. This experiment separately drops the title, drops the body or drops the whole terminal passage.

| Evidence | FLAN-T5-base exact / 160 | FLAN-T5-large exact / 160 |
| --- | ---: | ---: |
| Question only | 0 | 0 |
| First three gold passages | 1 | 1 |
| Terminal title | 3 | 2 |
| Terminal body | 69 | 98 |
| Terminal title and body | 68 | 91 |
| All four gold passages | 73 | 91 |

In a second control, 60 of 160 cases had a unique written answer alias in the visible last passage with no literal alias in the question or the earlier passage windows, plus a different donor string from the cohort with the same token count and coarse surface form. The matching rules and deterministic donor assignment run before reading model outputs. Only the matching answer string in the terminal text changes. This artificial edit can make the passage factually inconsistent.

| Reader and evidence | Original exact before edit / 60 | Original exact after edit / 60 | Donor exact after edit / 60 | Followed edit among initially exact |
| --- | ---: | ---: | ---: | ---: |
| Base, four gold | 42 | 0 | 32 | 32 / 42 |
| Base, terminal | 33 | 0 | 27 | 26 / 33 |
| Large, four gold | 43 | 0 | 28 | 28 / 43 |
| Large, terminal | 45 | 0 | 30 | 30 / 45 |

Answer accuracy in this control depends strongly on the terminal body. Omitting or replacing it changes the reader's answer even with the earlier annotated evidence held fixed. The shaped donor subset favors examples with identifiable answer strings and does not measure arbitrary counterfactual coherence. The reader comparison alone leaves open whether a retrieval policy can find the correct terminal passage among plausible competing endpoints. Continue to report annotated support recovery and final exactness jointly for the main benchmark.

HPG L4 array 53014140. [Paired numerical outcomes and controls](../../results/nonlinear_retrieval_natural/musique_terminal_evidence_v2/) · [Passage and reader budgets](MUSIQUE_READER_WINDOWS.md)
