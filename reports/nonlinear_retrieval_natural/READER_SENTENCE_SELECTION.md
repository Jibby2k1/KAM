# Automatic sentence selection with fixed retrieved evidence

The full article prefix and the companion-title sentence selector each produced **369 joint exact matches** on 1,024 HotpotQA questions. Query-only sentence selection produced **367**, and the two lead sentences produced **336**. Automatic selection recovered much of the short-context loss. It did not reproduce the [gold-sentence oracle](GOLD_READER_ORACLE.md) improvement over the full prefix.

## Design

We held fixed the article IDs and order from the [broad-ranked anchor study](BROAD_RANKED_ANCHOR.md). The FLAN-T5-small checkpoint, prompt format, 192-token per-article cap, and generation procedure matched that study. The full-prefix control reproduced all 1,024 archived exactness labels on four HPG L4 shards. All four policies use the same 560 recovered gold pairs.

The two short dense policies score the first 64 sentence candidates per retrieved article using frozen E5-small. Each returns the top two in article order. The query-only policy uses the original question. The companion-title policy appends the title of the other retrieved article. The two lead sentences serve as a short-input control. Gold annotations were used for visibility and joint scoring only.

## Results

| Reader evidence | Exact matches | Joint exact match | Visible annotations |
| --- | ---: | ---: | ---: |
| Full prefix | 445 | 369 | 1,637 |
| First two sentences | 401 | 336 | 1,270 |
| Query-only dense top two | 444 | 367 | 1,451 |
| Companion-title dense top two | 445 | 369 | 1,459 |

Query-only selection gained **3.03 percentage points** in joint EM over the two lead sentences (paired 95% interval **+1.56 to +4.49**). Relative to the full prefix it lost 0.20 points (interval **−1.56 to +1.17**). The companion-title selector gained 24 and lost 24 joint matches against the prefix (interval **−1.37 to +1.37**). The retrospective oracle choosing among the full prefix and both dense variants reached 396 joint matches.

On the retrieved-gold slice, the full prefix retained 1,255 of 1,359 valid annotated sentences. Query-only and companion-title selection retained 1,117 and 1,130. This visibility audit uses normalized lexical matching in the reader input. The results support retaining the prefix as the accuracy reference and investigating adaptive sentence budgets if short-input efficiency matters. Timing was outside the test design. These experiments reused fixed retrieved pairs, with no reader fitting.

## Reproduce

All four HPG **52952278** L4 shards completed. The [replay and extracted visibility annotations](../../results/nonlinear_retrieval_natural/reader_selection_v1/) and [Slurm driver](../../slurm/nonlinear_retrieval_reader_selection.sbatch) are available.

```bash
base=results/nonlinear_retrieval_natural
study="$base/reader_selection_v1"
anchor="$base/broad_anchor_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_reader_sentence_selection.py "$study/replay.json.gz" \
  --fixture "$anchor/fixture.json.gz" --reference "$anchor/replay.json.gz" --evidence "$study/gold-evidence.json.gz" \
  --output /tmp/kam-reader-selection.json
diff -u "$study/summary.json" /tmp/kam-reader-selection.json
```
