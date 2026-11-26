# IIRC evidence visibility after the correct articles are found

With both annotated linked articles supplied, their first 160 words exposed **all linked evidence on 98 of 139** fully aligned IIRC questions. A question-only BM25 window exposed all linked evidence on **9**, and an annotation-informed window at the same budget could expose it on **134**. A separate 100-word limit on the original source paragraph reduced complete three-input visibility further. [Visibility audit](../../results/nonlinear_retrieval_natural/iirc_visibility_v1/summary.json)

| Text budget and criterion | First window | Question-only BM25 | Annotation-informed window |
| --- | ---: | ---: | ---: |
| At least one annotated span from each linked article | 102 | 10 | 139 |
| All annotated spans from both linked articles | 98 | 9 | 134 |
| All annotated spans across both links and the source | 66 | 7 | 89 |
| Individual linked spans retained, of 288 | 227 | 66 | 283 |

For complete linked evidence, BM25 had one paired win and 90 losses against the lead. The change was **−64.0 percentage points** with a **−72.0 to −55.7** paired 95% interval from resampling source articles. The first 100 source words exposed every annotated source span on **92 of 139** questions even with the linked inputs held aside. None of these metrics use a reader or predict an answer. [Per-question counts](../../results/nonlinear_retrieval_natural/iirc_visibility_v1/)

## Matched visibility audit

We use the 153 official IIRC development questions from the frozen two-link routing study. The source paragraph and both linked gold IDs are given to every arm. Each linked article gets one 160-word window. The grid advances by 80 words and adds a final window ending at the last word. BM25 ranks those windows against the question alone, with document-local window statistics, a fixed stop list, and earliest-position tie breaking. The annotation-informed control chooses the window containing the most complete aligned spans, also breaking ties by earliest position. The source always uses its first 100 words. These word caps diagnose the text budgets used before tokenization. [Frozen routing cohort](IIRC_PAIR_TRANSFER.md) · [Extraction code](../../scripts/audit_kam_iirc_visibility.py)

The published offsets aligned exactly for all 212 original-source annotations and 301 of 318 linked annotations. The audit accepts an offset match or a verbatim quote with a unique occurrence in the same article. Seventeen linked annotations remained unresolved. Their 14 questions are included in the alignment ledger and excluded from the matched visibility table. This leaves 139 questions and 111 source articles for the comparison.

The dataset authors report that question writers saw link titles without the linked text, which makes lexical matching to the answer-bearing context difficult. Here a simple BM25 window scorer lost most of the available evidence relative to the lead. A selector trained with source-link context could target the gap to the annotation-informed control. The first reader experiment should preserve the lead baseline and give the original paragraph its own budget. [IIRC paper](https://aclanthology.org/2020.emnlp-main.86/)

## Reproduce

Recreate the counts with the official development JSON and linked archive from the [authors' setup](https://raw.githubusercontent.com/jferguson144/IIRC-baseline/main/setup.sh). The extraction requires `ijson`. The saved replay contains counts and window indices.

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_visibility_v1"
manifest="$base/iirc_pair_v1/manifest.json"
python scripts/audit_kam_iirc_visibility.py --dev iirc/dev.json --articles context_articles.tar.gz \
  --manifest "$manifest" --output /tmp/kam-iirc-visibility.json
gzip -cd "$study/replay.json.gz" | diff /tmp/kam-iirc-visibility.json -
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_iirc_visibility.py "$study/replay.json.gz" \
  --manifest "$manifest" --output /tmp/kam-iirc-visibility-summary.json
diff -u "$study/summary.json" /tmp/kam-iirc-visibility-summary.json
```
