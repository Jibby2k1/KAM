# Reader size at intermediate MuSiQue bindings

Changing FLAN-T5-base to FLAN-T5-large at the first intermediate hop improved first-binding exact match in every depth group. Replacing every intermediate reader reduced four-hop complete candidate pools from 52 to 40 and joint successes from seven to six. The observed effects depend on where the reader changes.

Across 576 author-template questions, all baseline records replayed exactly. The tokenizer vocabulary, evidence text cap, top-one paragraph rule and reader prompt stay fixed at each switched step. A changed intermediate prediction can affect the next step's retrieval. Changing only the final reader leaves the entire retrieval path fixed. Joint requires a complete 128-candidate pool and an exact answer to the final author subquestion.

| Four-hop policy, 160 questions | First binding exact | Penultimate binding exact | Complete pool | Last subanswer exact | Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base throughout | 65 | 37 | 52 | 12 | 7 |
| Large at first | 77 | 43 | 56 | 14 | 10 |
| Large at penultimate | 65 | 29 | 50 | 12 | 7 |
| Large at all intermediate steps | 77 | 33 | 40 | 10 | 6 |
| Large at final answer | 65 | 37 | 52 | 16 | 9 |

All four-hop joint intervals touched or included zero. At the penultimate step the larger reader saw exactly the same paragraphs, including 42 containing gold evidence, and answered fewer bindings exactly. A gold-binding intervention at the same step previously raised joint outcomes from 7 to 21. This preserves a diagnosis of missed late grounding while reducing the evidence for a larger general reader as a remedy.

HPG array 52990226. [Paired evidence and stage outcomes](../../results/nonlinear_retrieval_natural/native_musique_reader_size_v1/) · [Gold-binding interventions](MUSIQUE_SINGLE_BINDING.md) · [Large reader weights](https://huggingface.co/google/flan-t5-large)
