# Native 2Wiki conditioned retrieval and fixed-article reading

Adding 64 conditioned documents to the initial dense top 64 recovered both required articles on 826 of 1,024 questions. Taking the unconditioned dense top 128 recovered 625. The paired increase was 19.63 percentage points, with a question-bootstrap interval of +17.19 to +22.17.

## Frozen evidence cohorts

The study matched 512 cases with every annotated sentence present in the index to 512 with both article titles present and at least one annotation absent. Matching used question type and annotation count. All 2,048 required article titles were distinct and excluded from the 3,912 reserved titles across earlier 2Wiki studies. The corresponding 1,956 reserved question IDs and normalized wordings were also excluded. The exact/partial match is determined using the corpus audit before scoring. An independently rerun cohort builder reproduced the identical fixture. [Corpus audit](CROSS_DATASET_WIKIPEDIA.md)

| Required articles recovered at depth 128 | Exact evidence, 512 | Partial evidence, 512 | All, 1,024 |
| --- | ---: | ---: | ---: |
| BM25 | 302 | 286 | 588 |
| Dense | 322 | 303 | 625 |
| Hybrid reciprocal rank fusion | 328 | 314 | 642 |
| Dense 64 plus conditioned 64 | 441 | 385 | 826 |

The conditioned arm preserves the original dense top 64, uses its first article solely as context for a second question encoding, and fills 64 unique additions from the blended original and conditioned vector. All methods used the same corpus and final candidate capacity. The result is title-level pool coverage. On the 512 exact-evidence cases it also counts complete annotated corpus text in the pool. The compared policies use different amounts of search work.

Conditioned expansion improved coverage by 119 of 512 exact-evidence cases (+23.24 points) and 82 of 512 partial-evidence cases (+16.02 points). Within the exact-evidence cohort, 208 questions started with an incomplete dense pool. On the 189 whose first hit was a required article, unconditioned and conditioned expansion completed 17 and 132 pools. On the 19 whose first hit was some other article, the counts were one and five. For the partial cohort, the corresponding counts were 14 versus 97 of 190 and three versus two of 36. First-hit correctness was used to analyze the results after scoring. [Ranked-pool replay and paired counts](../../results/nonlinear_retrieval_natural/native_twowiki_v1/)

## Same-article reader controls

A frozen reader received both required articles in every arm, in the same case-specific order. The 192-token per-article cap and 512-token input cap applied throughout. The exact-match counts below use source answer strings. Annotation visibility uses the same tokenizer to normalize both the reference sentences and the submitted evidence before checking contiguous tokenized-word matches.

| Reader evidence | Exact cohort EM | Exact cohort all annotations visible | Partial cohort EM | Partial cohort all annotations visible |
| --- | ---: | ---: | ---: | ---: |
| Native indexed leads | 185 | 454 | 135 | 1 |
| Dataset-supplied article text | 179 | 452 | 155 | 402 |
| Annotated sentences only | 208 | 461 | 199 | 454 |

On the partial cohort, supplying the dataset text improved EM by 3.91 points (paired interval +0.78 to +7.03), and using just annotated sentences improved EM by 12.50 points (+8.79 to +16.21). On the exact cohort the annotation-only gain was 4.49 points (+1.56 to +7.62). All annotation-only inputs contained the full normalized text before the prompt limits. The prompt audit reproduced the original reader-side visibility flags and attributed additional literal misses to tokenizer normalization and truncation.

HPG search arrays 52967945 and 52968757 produced the sealed pools. Array 52968314 ran the fixed-article reader, and CPU job 52969359 audited prompt visibility. The report contains candidate-pool and fixed-article controls. A separate experiment transfers the locked HotpotQA pair scorer to the frozen lists. [Search runner](../../scripts/run_kam_native_twowiki.py) · [Reader runner](../../scripts/run_kam_native_twowiki_reader.py) · [Prompt audit](../../scripts/audit_kam_native_twowiki_prompts.py)
