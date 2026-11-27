# Matched decomposer capacities on MuSiQue

FLAN-T5-base decomposition recovered complete paragraph pools for 284–289 of 576 questions across three seeds, compared with 264–266 for FLAN-T5-small. Four-hop recovery rose from 16–21 to 25–29 of 160. Those four-hop contrasts were positive in every seed, with all paired intervals spanning zero.

Both models train on the same disjoint author-decomposition examples, with the same 480 optimizer steps and six examples per hop depth per update. The reference markers use the original textual representation. Training and plan decoding use B200; the same frozen E5 corpus search and FLAN-T5-base intermediate reader apply to both sets of plans on L4. Invalid plans fall back to the same hybrid result. The original question is the only test-time input to the planner. Pools are capped at the same 128 paragraphs, and fitting and monitoring examples follow the original title-disjoint protocol.

| Question depth | Questions | Hybrid support | Small support | Base support |
| --- | ---: | ---: | ---: | ---: |
| Two | 224 | 135 | 161–164 | 162–166 |
| Three | 192 | 56 | 83–85 | 92–99 |
| Four | 160 | 12 | 16–21 | 25–29 |
| Total | 576 | 203 | 264–266 | 284–289 |

On all questions, paired base-minus-small gains were 3.30, 3.99 and 3.99 percentage points. Shared-evidence cluster intervals excluded zero in two of the three seeds. Author dependency agreement improved from 240–268 to 291–317 overall, driven mainly by two-hop plans. Agreement on four-hop plans did not improve consistently. Base has 248 million parameters versus 77 million for small. The B200 optimizer-loop medians were 32.4 and 22.2 seconds. These results compare candidates and structural quality. Whether complete evidence reaches a four-paragraph reader and yields a correct answer is a separate readout.

HPG arrays 52988786 and 52988787. [Fits, ranks and paired analysis](../../results/nonlinear_retrieval_natural/native_musique_decomposer_scale_v1/) · [Original decomposer](MUSIQUE_LEARNED_DECOMPOSITION.md) · [Checkpoint family](https://huggingface.co/google/flan-t5-base)
