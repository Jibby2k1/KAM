# Atomic backward references in learned MuSiQue decomposition

Encoding dependency indices as single tokens made it possible to forbid forward references during decoding. This improved plan validity for the same checkpoint but made little difference to four-hop evidence recovery. The generated dependency graphs agreed less often with the author graphs than the original text-reference baseline.

The three atomic-token checkpoints warm-start from the same FLAN-T5-small model and use the same 3,584 fit questions, seed-specific batches, equal hop sampling, learning rate, steps, monitor and held-out panel as the text-form models. This changes the token representation of the references. At inference the optional mask permits a sentinel only if the referenced earlier segment is complete. Both atomic arms recover ordinary text references before the same corpus search and frozen binding reader. Candidate capacity remains 128. Author graphs are used to score structure after decoding.

| Result | Text references | Atomic references | Atomic with backward mask |
| --- | ---: | ---: | ---: |
| Valid plans, out of 576 | 504 / 503 / 517 | 493 / 512 / 510 | 517 / 549 / 550 |
| Complete support, out of 576 | 263 / 266 / 261 | 255 / 256 / 251 | 258 / 257 / 254 |
| Correct number of segments, four-hop subset of 160 | 37 / 46 / 36 | 57 / 49 / 32 | 57 / 49 / 32 |
| Exact author dependency graph, four-hop subset | 14 / 27 / 9 | 1 / 2 / 4 | 1 / 2 / 4 |
| Complete support, four-hop subset | 18 / 17 / 13 | 14 / 14 / 17 | 14 / 14 / 18 |

The hybrid baseline found complete support on 203 overall and 12 four-hop questions. A valid grammar rules out invalid connections. The policy must still predict which earlier answer belongs to each subquestion. These results motivate improving subquestion and dependency prediction together before adding a stronger structural mask. The 576-question panel was used for earlier diagnostics. Treat the count and graph comparisons as conditional on that panel.

HPG array 52986262 trained three seeds and produced both plans from the atomic checkpoints. Array 52986263 replayed the frozen corpus comparison. [Counts, numeric rankings and provenance](../../results/nonlinear_retrieval_natural/native_musique_atomic_bindings_v1/) · [Original decomposer](MUSIQUE_LEARNED_DECOMPOSITION.md) · [Known depth comparison](MUSIQUE_DEPTH_CONTROL.md)
