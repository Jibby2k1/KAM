# Single-binding interventions on MuSiQue

On 160 four-hop questions with author-supplied subquestion templates, replacing the first intermediate answer recovered 23 additional complete candidate pools and two additional joint successes. Replacing the third intermediate answer recovered 14 additional complete pools and 14 additional joint successes. Early and late binding errors can therefore have different impacts even in the same pipeline.

All 576 questions reuse the established native MuSiQue cohort and fixed 64 initial plus 64 expansion budget. Each intervention replaces one predicted intermediate answer with its labeled value solely when filling later subquestions. The sentence encoder, candidate retrieval, top-one paragraph reader, and previous generated answers are held fixed through the binding insertion. The scored answer at the intervened step remains the reader's original prediction. Subsequent nonintervened bindings are predicted normally. All 576 top-one baseline records replayed exactly. Joint here means a complete 128-candidate pool and exact answer to the last author subquestion. It is a diagnostic distinct from joint success after final paragraph selection on the original question.

| Questions | Injected binding | All supports in pool | Last subanswer exact | Joint |
| --- | --- | ---: | ---: | ---: |
| Two hops, 224 | None | 183 | 52 | 50 |
| Two hops, 224 | First | 204 | 78 | 75 |
| Three hops, 192 | None | 120 | 30 | 27 |
| Three hops, 192 | First | 144 | 35 | 33 |
| Three hops, 192 | Second | 146 | 41 | 35 |
| Four hops, 160 | None | 52 | 12 | 7 |
| Four hops, 160 | First | 75 | 13 | 9 |
| Four hops, 160 | Second | 60 | 17 | 8 |
| Four hops, 160 | Third | 66 | 42 | 21 |

For two hops, oracle correction added 11.16 percentage points joint (95% shared-evidence cluster interval 6.88 to 15.91). For four hops, the first correction added 14.38 points complete support (5.24 to 24.85). The third added 8.75 points joint (1.85 to 16.67) and placed the final supporting paragraph first for 91 questions versus 38. Because author templates and one gold answer enter the intervention, these gains bound the tested failure mode under supplied structure. They motivate targeted tests of intermediate grounding and the final-reader interface.

HPG array 52988909. [Matched outcomes](../../results/nonlinear_retrieval_natural/native_musique_single_binding_v1/) · [Runner](../../scripts/run_kam_musique_bindings.py) · [Analyzer](../../scripts/analyze_kam_musique_single_binding.py) · [Baseline binding study](MUSIQUE_BINDINGS.md)
