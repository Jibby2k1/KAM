# Reader sensitivity to document order

Reversing the retrieved evidence pair changed 76 jointly scored outcomes, with 40 wins and 36 losses. Joint EM rose from **369 to 373** on 1,024 questions, a paired **+0.39 percentage points** (95% interval **−1.27 to +2.15**). The interval includes zero. The experiment does not establish an advantage for switching the default order.

## Design and results

The [broad-ranked anchor replay](BROAD_RANKED_ANCHOR.md) fixed each retrieved evidence pair. We evaluated the archived order and its reversal using the same document text, per-document 192-token prefix, prompt, frozen reader, and normalized answer comparison. The archived order agreed with the original reader run on all 1,024 exactness labels. Neither policy uses gold evidence at inference. The full cohort retains the same 560 gold support pairs.

| Fixed-pair scope | Questions | Archived exact | Reversed exact | Archived joint EM | Reversed joint EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 1,024 | 445 | 455 | 369 | 373 |
| Both supports retrieved | 560 | 369 | 373 | 369 | 373 |
| Support missing | 464 | 76 | 82 | 0 | 0 |

The either-order oracle produced 409 joint matches. That range establishes order sensitivity on individual questions while leaving the choice of order unresolved. This experiment holds retrieval and sentence selection fixed and measures only the reader.

## Reproduce

HPG job **52952705** ran four L4 shards. The [ordered input audit and replay](../../results/nonlinear_retrieval_natural/reader_order_v1/) and [Slurm driver](../../slurm/nonlinear_retrieval_reader_order.sbatch) are available.

```bash
base=results/nonlinear_retrieval_natural
study="$base/reader_order_v1"
anchor="$base/broad_anchor_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_reader_order.py "$study/replay.json.gz" \
  --fixture "$anchor/fixture.json.gz" --reference "$anchor/replay.json.gz" --output /tmp/kam-reader-order.json
diff -u "$study/summary.json" /tmp/kam-reader-order.json
```
