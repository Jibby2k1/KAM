# Learned question decomposition on a frozen MuSiQue paragraph index

An inference-only learned decomposer recovered complete evidence on 261–266 of 576 held-out MuSiQue questions across three independent fits. Hybrid lexical and dense retrieval recovered 203 at the same 128-paragraph capacity. The increase was concentrated in two- and three-hop questions. Four-hop coverage remained far below the control with author templates.

## Fit and inference separation

The small FLAN-T5 checkpoint was fine-tuned to predict sequences of subquestions and backward references from just the original question. Targets contained author subquestion text and placeholders. They excluded answer strings and paragraph text. Fit used 3,584 questions from the official training split, with 2,048 two-hop, 1,152 three-hop, and 384 four-hop examples. It used 480 updates and six examples per hop per update. The independent format-monitoring split had 208 questions. Fit, monitoring, and the sealed test shared no required article titles or normalized question strings. The monitor deliberately reserved less-shared support titles to preserve scarce four-hop training examples.

At inference the system generated a variable number of subquestions. Only structurally valid plans with two to four steps and backward references entered the route. Invalid plans fell back to the archived hybrid ranking. For a valid plan, the first 64 candidates came from the frozen dense search. The remaining 64 slots were divided across the predicted number of steps. A frozen FLAN-T5-base reader supplied each needed intermediate value from that subquery's top paragraph. Neither author templates nor true graph depth entered the learned route on test. [Native corpus and frozen questions](NATIVE_MUSIQUE.md) · [Author-template control](MUSIQUE_BINDINGS.md)

## Complete support at matched depth

| Policy | 2 hops, 224 | 3 hops, 192 | 4 hops, 160 | All, 576 |
| --- | ---: | ---: | ---: | ---: |
| Hybrid lexical and dense | 135 | 56 | 12 | 203 |
| Learned seed 106331 | 159 | 86 | 18 | 263 |
| Learned seed 106339 | 169 | 80 | 17 | 266 |
| Learned seed 106367 | 163 | 85 | 13 | 261 |
| Author templates with predicted bindings | 183 | 120 | 52 | 355 |
| Author templates with gold bindings | 204 | 164 | 95 | 463 |

Overall, the learned route gained 10.07–10.94 percentage points over hybrid. The three paired shared-evidence cluster bootstrap intervals were +5.99 to +14.77, +6.66 to +15.54, and +5.59 to +14.68. The gains in four-hop questions ranged from one to six completed chains, with all three interval lower bounds at or below zero. The learned route executes up to four dense searches and three intermediate generations. This comparison matches candidate capacity and does not match latency.

The base checkpoint before fitting generated zero plans passing the parser on the held-out set. The fitted models passed 504, 503, and 517 of 576. On four-hop questions they predicted exactly four steps for 37, 46, and 36 of 160, and matched the complete author reference signature for 12, 26, and 8. Most four-hop questions were under-decomposed to three steps. This distinction separates syntactic validity from recovery of the underlying evidence chain.

HPG L4 array 52976152 fit the three seeds. Arrays 52976435 and 52976436 generated frozen plans from the fit and zero-shot checkpoints. Arrays 52976459, 52976460, and 52976461 retrieved and read intermediate evidence from independently frozen plans. Public artifacts include numeric candidate IDs, training and test hashes, support counts, and split manifests. They record retrieval coverage. [Numeric replay and training manifests](../../results/nonlinear_retrieval_natural/musique_learned_decomposition_v1/) · [Fit runner](../../scripts/train_kam_musique_decomposer.py) · [Search runner](../../scripts/run_kam_musique_learned_search.py)
