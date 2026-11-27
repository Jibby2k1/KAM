# Binary scholarly evidence on QASPER

A frozen FLAN-T5-base reader answered 51 of 74 held-out binary questions correctly when it received annotator-marked evidence, compared with 34 using nonannotated paragraphs from the same paper. With dense top-two paragraphs it answered 47 correctly, and 15 of those answers had complete annotated support.

Seventy-four test questions from 65 papers are disjoint from the earlier pilot question set. Each has at least two agreeing binary annotations, no conflicting binary vote and an aligned annotated supporting set of at most two paragraphs. The gold condition uses an aligned annotation set. Within-paper E5 ranks paragraphs using the same fixed two-paragraph upper bound as the sham. Annotated paragraphs are excluded from the sham. No answer labels were used to choose dense results. The 48 yes and 26 no labels give an always-yes baseline of 48/74.

| Prompt evidence | Yes answers correct | No answers correct | All correct | Complete support | Joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Question only | 4 | 24 | 28 | 0 | 0 |
| Same-paper sham | 12 | 22 | 34 | 0 | 0 |
| Within-paper dense top two | 29 | 18 | 47 | 26 | 15 |
| Annotated support | 33 | 18 | 51 | 74 | 51 |

Greedy generation and forced binary scoring produced the same per-arm aggregate counts. At the question level, annotated evidence increased accuracy by 23.0 percentage points versus same-paper sham (95% paper-cluster interval 9.5 to 36.8) and by 31.1 points versus question only (16.7 to 45.3). Dense versus annotated answer accuracy was −5.4 points (−15.6 to 5.6). The similar aggregate dense and gold accuracies coexist with a substantial gap in annotated support. Nonannotated paragraphs can still contain useful signals. The comparison covers the binary slice with compact textual annotations.

HPG array 52988530. [Question manifest, scores and paired contrasts](../../results/nonlinear_retrieval_natural/qasper_binary_v1/) · [Fixture builder](../../scripts/prepare_kam_qasper_binary.py) · [Runner](../../scripts/run_kam_qasper_binary.py) · [Source benchmark](https://aclanthology.org/2021.naacl-main.365/)
