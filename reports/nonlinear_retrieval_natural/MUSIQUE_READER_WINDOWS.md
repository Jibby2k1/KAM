# Four-hop MuSiQue reader capacity and passage controls

A wider passage window improved final-answer exact match with all four gold paragraphs. The same large reader reached 91 of 160 exact answers on the final gold paragraph alone and on all four gold paragraphs. Those results shared 78 successes. The final paragraph is identified by the author’s decomposition, so this control probes the reader’s dependence on earlier evidence.

The same 160 four-hop questions from the fixed composition study use the original question. Four HPG L4 jobs compare FLAN-T5-base and FLAN-T5-large on identical input token IDs. Each receives the same archived four gold paragraphs or the same four actually selected paragraphs from the original greedy bindings. Reader prompts vary by per-passage capacity and by which evidence is presented. The prefix/center controls keep up to 32 title tokens, then allocate the remaining budget to the passage beginning or the fixed question-token-overlap window. Trials use FP32. All 320 archived base outcomes replayed exactly.

| Passage policy | Gold exact, base / large | Selected exact, base / large | Selected joint, base / large |
| --- | ---: | ---: | ---: |
| Original 96 | 59 / 76 | 22 / 26 | 5 / 8 |
| Title and prefix, 96 | 59 / 76 | 22 / 26 | 5 / 8 |
| Title and centered body, 96 | 59 / 74 | 21 / 25 | 5 / 8 |
| Title and prefix, 192 | 73 / 91 | 24 / 30 | 5 / 8 |
| Title and centered body, 192 | 73 / 89 | 24 / 30 | 5 / 8 |
| First hop alone, 192 | 0 / 1 | 0 / 1 | 0 / 0 |
| Final hop alone, 192 | 68 / 91 | 23 / 26 | 0 / 0 |

Among gold-only prompts, the 192-prefix gain over the original 96 was +8.75 percentage points [0.85, 17.60] for base and +9.38 [3.01, 16.09] for large. The 192-prefix large-versus-base interval crosses zero [−3.09, 27.43]. Only nine of the actually selected four-paragraph sets contained all gold paragraphs. At fixed reader size, wider windows gave no extra joint outcomes. The intervals resample the 27 shared-evidence clusters.

The final-only and four-passage large settings each scored 91, with 13 answers gained and 13 lost. The paired interval for the final-only change is [−9.70, +9.77] points. The last-only control for the selected panel uses the last greedy assignment slot. Joint evidence coverage requires all four gold paragraph IDs, so single-passage arms have zero joint recovery by construction. The two readers use different parameter counts. On gold and selected prompts combined, widening archived 96 to the 192 prefix changed summed generation time from 7.03 to 10.15 seconds for base and from 21.34 to 30.64 seconds for large. Timing covers pretokenized generation and excludes model load and data preparation.

HPG L4 array 53012831. [Audit and paired outcomes](../../results/nonlinear_retrieval_natural/musique_reader_window_v3/) · [Fixed composition study](MUSIQUE_FIXED_COMPOSITION.md)
