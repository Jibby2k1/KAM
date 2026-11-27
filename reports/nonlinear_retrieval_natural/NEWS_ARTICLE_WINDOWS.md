# Fixed-article evidence windows for MultiHop-RAG news

With the two annotated news articles supplied, selecting one question-matched sentence window per article increased joint exact match from 16 to 31 of 384 questions relative to word-stride windows. Annotated fact prompts reached 241 joint matches, establishing a large remaining gap in the news reader pipeline.

The 384 questions are disjoint from an earlier pilot and balanced across temporal, comparison and inference categories. Article URLs can recur across questions and pilots. This experiment fits no model. Each fully aligned question uses two annotated articles, the same frozen FLAN-T5-base reader and a 512-token total prompt ceiling. The article cap is 192 tokens per article. Windows are packed in exact reader token IDs. Joint requires exact answer and visibility of every annotated fact in the prompt as measured on normalized spans.

| Evidence selection | All facts visible | Exact answer | Joint exact |
| --- | ---: | ---: | ---: |
| No evidence | 0 | 141 | 0 |
| Article lead | 7 | 103 | 4 |
| Word-stride windows | 46 | 111 | 16 |
| One sentence center per article | 69 | 134 | 31 |
| Two sentence centers per article | 89 | 118 | 33 |
| Annotated facts | 381 | 243 | 241 |

One center improved joint exact by 3.91 percentage points relative to stride (95% shared-article cluster interval 1.82 to 7.14). Two centers recovered 20 additional complete fact sets and added two net joint successes relative to one center. The paired interval spans zero. A cluster of 117 questions shares evidence links transitively, so effective replication is below the nominal question count. Question-only exact accuracy is high in this subset and increases the value of reporting evidence-grounded joint correctness.

HPG array 52989090. [Question manifest and paired contrasts](../../results/nonlinear_retrieval_natural/news_windows_v1/) · [Runner](../../scripts/run_kam_news_windows.py) · [Source benchmark](https://github.com/yixuantt/MultiHop-RAG)
