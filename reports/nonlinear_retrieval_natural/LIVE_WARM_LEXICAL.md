# Exact threaded hybrid retrieval after lexical prewarm

With the lexical and article files prewarmed on HPG B200 nodes, two segment threads saved **4.64 ms** at the paired median full request versus warmed serial hybrid (95% question-bootstrap interval **0.91–10.55 ms**). The threaded route preserved the initial ranks, live anchors, expansion order, guard features and both reader decisions on all 128 questions. [Paired timing and parity](../../results/nonlinear_retrieval_natural/live_warm_lexical_v1/summary.json)

| Initial route at depth 64 | Request p50 / p95 | Initial-search p50 / p95 | Both supports selected | Joint EM |
| --- | ---: | ---: | ---: | ---: |
| Exact dense | 75.01 / 102.77 ms | 2.94 / 2.94 ms | 66 | 45 |
| Exact hybrid, warmed serial | 124.12 / 175.13 ms | 49.35 / 84.93 ms | 67 | 45 |
| Exact hybrid, warmed two threads | 111.53 / 173.75 ms | 41.80 / 86.13 ms | 67 | 45 |

The paired initial-search saving from two threads was **2.11 ms** (**0.61–11.11 ms**). With two threads, hybrid still added **38.12 ms** paired median over dense (**34.37–41.45 ms**). Search, article fetch, the live broad-ranked anchor, expansion, reranking with sentence reuse, and the reader were included in the full clock. Benchmark p95 changes should be interpreted on this 128-question sample. [Raw traces](../../results/nonlinear_retrieval_natural/live_warm_lexical_v1/)

## Matched protocol

HPG array **52959320** used four B200 workers and the same sealed 128 questions as the earlier dense–hybrid timing sample. Each read the article database and the 380 MB lexical files sequentially before four separate warmup requests per policy. The lexical read took 0.08–0.28 seconds. Each worker then ran the three paths in seeded randomized order for each of its 32 questions. Only the Tantivy segment executor differed between the two warmed hybrid policies. Query parsing, depth 128 in each lexical and dense ranking, rank-fusion configuration, and downstream model weights stayed fixed. [HPG driver](../../slurm/nonlinear_retrieval_live_anchor_profile.sbatch) · [Isolated CPU test](LEXICAL_PREWARM.md)

All three paths reproduced the original top-64 order and all selected outcomes on every query relative to the older quality replay. The 64 expanded candidates matched membership on all questions, and order matched on 123. The two warmed hybrid paths agreed exactly with each other on all 128 expanded orders and outcomes. Their full p50 values are lower than the earlier first-pass hybrid run's 229.05 ms. That earlier run used different allocations and ordinary lexical warmups. Within this study the threaded comparison isolates the executor after warming. [Matched first-pass timing](LIVE_DENSE_HYBRID_TIMING.md) · [Quality study](DENSE_HYBRID_ANCHOR.md)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_warm_lexical_v1"
python scripts/analyze_kam_live_initial_profile.py "$study"/profile-52959320-*.json.gz \
  --warm-lexical --study "$base/dense_hybrid_anchor_v1" --output /tmp/kam-warm-lexical.json
diff -u "$study/summary.json" /tmp/kam-warm-lexical.json
```
