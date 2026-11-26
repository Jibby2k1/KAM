# Frozen reader accuracy with gold HotpotQA evidence

The frozen reader reached **611 exact matches in 1,024 questions** from indexed gold articles. With annotated supporting sentences it reached **660 in 1,023**. On the common valid subset, sentence-only EM improved by **4.79 percentage points** (paired 95% interval **+2.54 to +7.04**). This is headroom under oracle evidence, with no change to retrieval or training.

## Design

The fresh dense/hybrid cohort was sealed before scoring. The passage audit verified question, answer, article IDs, title normalization, source hashes, and sentence positions against the official training files on this held-out set. We supplied the same two gold article titles in canonical order to each arm. The three inputs were the indexed article text, the original dataset passage, and the dataset's annotated supporting sentences. The same frozen FLAN-T5-small reader, prompts, token budgets (192 per article, 512 for the concatenated prompt), and normalized exact-match measure applied to each. One annotation pointed beyond its source passage. Its question remained in the two full-evidence arms and was excluded from sentence-only comparisons.

## Results

| Evidence | Exact matches | Questions |
| --- | ---: | ---: |
| Indexed gold articles | 611 | 1,024 |
| Dataset gold passages | 610 | 1,024 |
| Annotated supporting sentences | 660 | 1,023 |

The indexed articles held 2,455 of 2,456 valid annotated sentences under normalized lexical matching, and **2,276** remained visible in the per-article reader prefix. On 868 questions with all valid sentences visible, indexed gold EM was 532 and sentence-only EM was 567. On 155 questions with at least one sentence missing from the prefix, the counts were 79 and 93. The diagnostic points to both truncation and competition from surrounding text. The two full gold arms agreed on 1,023 of 1,024 results. The remaining errors in the oracle sentence arm show that better retrieval and localization leave a separate reader limitation.

The test supplies gold support and uses their canonical order. It measures reader accuracy under those inputs and does not evaluate an automatic sentence selector or establish a ceiling across all orderings and prompts.

## Reproduce

All four HPG L4 shards of **52951864** completed. The [fixture, extracted evidence and replay](../../results/nonlinear_retrieval_natural/gold_reader_oracle_v1/) and [Slurm driver](../../slurm/nonlinear_retrieval_gold_reader_oracle.sbatch) record the frozen inputs and results.

```bash
study=results/nonlinear_retrieval_natural/gold_reader_oracle_v1
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_gold_reader_oracle.py "$study/replay.json.gz" \
  --fixture "$study/fixture.json.gz" --evidence "$study/gold-evidence.json.gz" --output /tmp/kam-reader-oracle.json
diff -u "$study/summary.json" /tmp/kam-reader-oracle.json
```
