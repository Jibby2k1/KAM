# Answer formatting with fixed IIRC evidence

A question-derived answer format raised the frozen FLAN-T5-small reader from **25 to 42 exact answers out of 134** with the original source and both gold linked article leads supplied. The paired gain was **+12.69 percentage points** (95% source-article interval **+5.93 to +19.85**, 20 wins and 3 losses). An oracle supplied with the annotated answer type reached 43. [Audited comparison](../../results/nonlinear_retrieval_natural/iirc_reader_format_v1/summary.json)

| Evidence | Answer type | Questions | Generic | Universal format | Question-derived type | Gold type |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Three leads | All | 134 | 25 | 35 | **42** | 43 |
| Three leads | Single-span | 90 | 24 | 20 | 24 | 25 |
| Three leads | Binary | 25 | 1 | 15 | **17** | 17 |
| Three leads | Numeric | 19 | 0 | 0 | 1 | 1 |
| All annotated spans | All | 134 | 38 | 45 | **47** | 48 |
| All annotated spans | Single-span | 90 | 35 | 30 | 30 | 31 |
| All annotated spans | Binary | 25 | 3 | 15 | **16** | 16 |
| All annotated spans | Numeric | 19 | 0 | 0 | 1 | 1 |

Most of the gain came from binary formatting. With three leads, the question-derived format produced a literal normalized yes/no on all 25 binary questions, versus two under the generic prompt. The question heuristic classified all 25 binary, 85 of 90 single-span, and 15 of 19 numeric questions correctly. The prompt yielded only one exact numeric answer on either evidence condition. With annotation spans, changing format improved the aggregate count and reduced single-span exactness by five. [Per-type outcomes](../../results/nonlinear_retrieval_natural/iirc_reader_format_v1/summary.json)

## What was held fixed

All policies use the pinned reader and the same questions, article identities, article order, and evidence text within each question and evidence condition. The universal instruction requests a phrase, yes/no, or digits. The question-derived policy dispatches on initial auxiliary verbs and fixed numeric question starters. Gold type is an oracle that uses annotation labels. There was no format tuning on this test cohort. This is answer decoding with article identities supplied. [Prior fixed-article diagnostic](IIRC_FIXED_ARTICLE_READER.md) · [Runner](../../scripts/run_kam_iirc_reader.py)

One longer instruction exceeded the 512-token limit in the first run. The revised comparison shortened the last evidence segment equally for all policies on that one lead query. Generic exactness matched the earlier run on all questions for both evidence conditions. The raw generic answer matched on 133 lead queries and all 134 annotation-span queries. A follow-up fixed the binary output counter to read normalized token lists. The final four L4 shards were **52957533**. [Raw-output fingerprint audit](../../results/nonlinear_retrieval_natural/iirc_reader_format_v1/)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_reader_format_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_iirc_reader_format.py \
  --replay "$study/replay.json.gz" --reference "$base/iirc_reader_v1/replay.json.gz" \
  --output /tmp/kam-iirc-format.json
diff -u "$study/summary.json" /tmp/kam-iirc-format.json
```
