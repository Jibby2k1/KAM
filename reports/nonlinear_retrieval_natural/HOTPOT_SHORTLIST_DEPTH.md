# Widening the sentence shortlist on fixed HotpotQA unions

Widening the frozen hybrid expansion shortlist from 16 to 64 proposals per head admitted the gold pair on **42 more of 1,024** questions. The final scorer recovered only **two** additional gold pairs overall and joint exact matches increased by **three**. At 32 proposals the system achieved the same support and joint counts as at 64. [Replayed counts](../../results/nonlinear_retrieval_natural/hybrid_shortlist_depth_v1/summary.json)

| Proposals per head | Gold pair shortlisted | Both supports selected | Joint EM | Mean distinct sentences requested |
| --- | ---: | ---: | ---: | ---: |
| 16 | 695 | 543 | 340 | 72.78 |
| 32 | 717 | 545 | 343 | 120.23 |
| 64 | 737 | 545 | 343 | 185.84 |

The joint change at 32 and 64 was **+0.29 percentage points** versus depth 16 (paired 95% question interval **0.00 to +0.68**). The support change was **+0.20 points** (**−0.20 to +0.59**). Within the 85 already-covered questions where the gold pair was absent at depth 16, widening to 64 admitted 42 gold pairs and the final model selected three, all with exact answers. The extra proposals also displaced one previously selected gold pair overall. [Paired audit](../../results/nonlinear_retrieval_natural/hybrid_shortlist_depth_v1/summary.json)

| Original hybrid pool | Questions | Support at 16 / 32 / 64 | Joint EM at 16 / 32 / 64 |
| --- | ---: | ---: | ---: |
| Complete | 659 | 474 / 476 / 476 | 296 / 299 / 299 |
| Incomplete | 365 | 69 / 69 / 69 | 44 / 44 / 44 |

## Matched replay

The diagnostic reused the sealed hybrid top-64 pools and ordered conditioned hits from the matched expansion study. All three arms scored the same candidate union with the same frozen shortlist heads, per-head score normalization over the same full pair universe, fitted sentence reranker, and FLAN-T5-small reader. The 64-deep pass supplied nested masks in stable rank order for the narrower budgets and shared one sentence encoding pass. The narrowest replay matched the archived B200 selected pair, shortlist label, and answer exactness on all 1,024 questions. HPG array **52957284** ran on four L4s. [Matched B200 source](DENSE_HYBRID_ANCHOR.md) · [Slurm driver](../../slurm/nonlinear_retrieval_hotpot_shortlist_depth.sbatch)

The actual mean pair counts were 26.90, 54.65, and 108.53 across depths 16, 32, and 64. At depth 64 the distinct sentence count was 2.55 times the depth-16 count. These are work counts for the requested candidate features. Runtime was shared across budgets. The higher shortlist size alone delivers little with the frozen final scorer. Keep the current budget as reference when evaluating a new final scorer on expanded pools. [Replay artifacts](../../results/nonlinear_retrieval_natural/hybrid_shortlist_depth_v1/)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/hybrid_shortlist_depth_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_hotpot_shortlist_depth.py \
  "$study"/replay-52957284-*.json.gz --study "$base/dense_hybrid_anchor_v1" \
  --output /tmp/kam-hotpot-shortlist.json
diff -u "$study/summary.json" /tmp/kam-hotpot-shortlist.json
```
