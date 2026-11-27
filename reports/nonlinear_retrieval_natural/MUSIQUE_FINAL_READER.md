# Final answer accuracy with learned MuSiQue evidence

Learned decomposition increased complete coverage in the 128-paragraph pool, but the four-paragraph final selector used little of that gain. On 576 held-out original questions, joint exact match was 18 for hybrid top four and 22–25 for the learned paths. A gold-aware selector restricted to each policy's candidates reached 74 with hybrid and 93–94 with the learned pools, exposing headroom in the final selection stage.

## Fixed reader and evidence budget

All retrieval policies used the same original question, FLAN-T5-base checkpoint, 96-token limit per paragraph, and 512-token prompt cap. The four-passage routes kept a common budget. Hybrid either used its own top ranks or reranked its 128 candidates by original-question dense similarity. The learned path retained the paragraphs used to read intermediate answers and the first newly added paragraph from the final subquery, deduplicated them, and filled any free slots from its frozen candidate order. An invalid learned plan used hybrid top four.

Gold-aware controls select the available annotations before filling remaining slots from the same frozen pool. The gold-only control supplies just the required paragraphs, and the question-only control has no context. The normalized exact match accepts the source answer and recorded aliases. Joint EM also requires every annotated paragraph among the submitted evidence. [Frozen learned pools](MUSIQUE_LEARNED_DECOMPOSITION.md)

| Original-question route, 576 questions | Complete final evidence | Answer EM | Joint EM |
| --- | ---: | ---: | ---: |
| Hybrid top four | 44 | 51 | 18 |
| Hybrid pool, original-question dense top four | 36 | 41 | 17 |
| Learned pools, original-question dense top four | 36 | 41 | 17 |
| Learned path, three seeds | 61–68 | 48–56 | 22–25 |
| Gold-aware selection within hybrid pool | 203 | 112 | 74 |
| Gold-aware selection within learned pools | 261–266 | 126–127 | 93–94 |
| Gold-only evidence | 576 | 231 | 231 |
| Question only | 0 | 4 | 0 |

The raw-question dense reranker proposed the same ordered evidence as on the hybrid pool for 575, 576, and 576 questions across the three learned seeds. Its small ranking scope rarely reached material exposed only by decomposition. The learned path gained four, four, and seven joint successes over hybrid, with all three shared-evidence paired intervals spanning zero. With gold-aware selection, the learned pools gained 19–20 joint successes and the three paired interval lower bounds were above zero. No deployable route recovered all four supports at this final budget on the 160 four-hop questions. Even the pure gold-paragraph control answered 59 of those 160 correctly.

HPG array 52978195 decoded 4,301 unique prompts across four L4 workers. Reusing identical prompts throughout each shard kept the reader outcomes paired. [Final evidence and paired results](../../results/nonlinear_retrieval_natural/musique_final_reader_v1/) · [Reader runner](../../scripts/run_kam_musique_final_reader.py)
