# Predicted intermediate answers under fixed MuSiQue templates

With author-provided subquestion templates, inserting reader-predicted intermediate answers increased four-hop complete support recovery from 13 to 52 of 160 questions. Giving the reader the true paragraph at every hop reached 64, and inserting the true intermediate answers reached 95. The 128-document candidate budget and first 64 documents were fixed throughout.

## Design

The 576 held-out questions and 101,748-paragraph bank match the [native MuSiQue study](NATIVE_MUSIQUE.md). The templates retain explicit backward references such as `#2`. At each step, only prior reader outputs fill those references. The frozen E5 retriever searches the complete corpus with the resulting subquestion. Each step contributes its fixed even share of 64 new unique paragraphs. A frozen FLAN-T5-base reader receives either the top paragraph, the top three, or, in a labeled oracle, the corresponding gold paragraph. Passages were capped at 128 tokens before generation. Gold answers in the top-one, top-three, and gold-paragraph reader policies were used only to score predictions. All three policies use privileged author templates. [Reader checkpoint](https://huggingface.co/google/flan-t5-base/tree/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2)

| Method | Full support, 2 hops | Full support, 3 hops | Full support, 4 hops | All, 576 |
| --- | ---: | ---: | ---: | ---: |
| Author templates, references removed | 140 | 72 | 13 | 225 |
| Predicted answers from top retrieved paragraph | 183 | 120 | 52 | 355 |
| Predicted answers from top three retrieved paragraphs | 167 | 109 | 42 | 318 |
| Predicted answers given gold paragraphs | 189 | 146 | 64 | 399 |
| Gold intermediate answers | 204 | 164 | 95 | 463 |

On four-hop questions, the improvement from reading the top paragraph was 24.38 percentage points (+10.75 to +37.84 shared-evidence cluster bootstrap interval), with 42 wins and three losses versus unresolved templates. The top-three reader recovered 10 fewer complete chains than top-one on this slice. It saw the annotated paragraph more often at every hop, while extracting fewer exact answers at hops two and three. This comparison changes both evidence volume and distractor exposure.

## Four-hop reading diagnostic

| Reading condition | Hop 1 answer EM | Hop 2 answer EM | Hop 3 answer EM | Hop 4 answer EM | All three intermediate answers exact | Full support and last subquestion EM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top one | 65 | 33 | 37 | 12 | 8 | 7 |
| Top three | 76 | 29 | 30 | 16 | 5 | 3 |
| Gold paragraph | 108 | 56 | 55 | 42 | 12 | 24 |

The last column scores the last authored subquestion, with its particular wording and references. It is separate from answer EM under the original undecomposed question. The gold-paragraph control still fed the reader its own earlier predictions. Its later-hop errors therefore include propagation from earlier answers. A reader can recover enough of an entity mention to improve ranking without matching the dataset's exact answer string at every intermediate hop.

HPG job 52974744 ran all four L4 shards. Both the template file and scoring labels were sealed from the authoritative development release and checked against the prior oracle audit. The public replay contains only case and paragraph IDs, ranks, and exactness flags. [Replay and analysis](../../results/nonlinear_retrieval_natural/musique_bindings_v1/) · [Runner](../../scripts/run_kam_musique_bindings.py)
