# Does the compact attention-mass certificate work on pretrained states?

The compact certificate verified no protected-shortlist rows at 90%, 95%, or 99% retained mass on pretrained SmolLM2. Exact attention showed 39.40% of the rows retained at least 95%. Applying the same moment bound directly to each query, without a tile-drift term, also certified none.

## Matched real-state audit

Use the pinned SmolLM2-135M weights and the same 16 PG-19 windows as the omitted-summary audit. The state comes from full causal attention. The protected route keeps eight 64-token blocks, shared across each 32-query tile for an individual head. Measure offsets 0, 8, 16 and 31 in 56 tiles per window, across all 30 layers and nine query heads. That gives 967,680 measured query-head rows. The exact and bounded log partitions use float64 on the same post-RoPE states. Every candidate upper partition bound exceeded the corresponding exact omitted-group partition within the numerical check tolerance. [Pinned checkpoint, matched books and scope](OMITTED_BLOCK_SUMMARIES.md)

The compact form stores mean keys, the largest residual norm and key norm, and a Gram row-sum bound per group. The tile version uses the initial query with a score-increase bound for query drift. The direct-query version recomputes the same bound for each query. The exact-interval control uses the true minimum, maximum and variance of individual group scores. It reads the omitted scores and measures how much of the gap can come from coarse score geometry alone. [Certificate derivation and controlled fixtures](SHORTLIST_MASS_CERTIFICATE.md)

| Bound | Rows passing 90% | Rows passing 95% | Rows passing 99% |
| --- | ---: | ---: | ---: |
| Actual retained mass | 482,859 (49.90%) | 381,228 (39.40%) | 188,970 (19.53%) |
| Compact representative with drift | 0 | 0 | 0 |
| Compact per-query | 0 | 0 | 0 |
| Exact per-group score interval and variance | 399,963 (41.33%) | 304,238 (31.44%) | 122,374 (12.65%) |

The exact-interval control certified 79.81% of rows that truly met 95%. The compact bound missed the requirement to certify half of such rows before a live fallback. It failed even on the first query of a tile, where the drift term is zero. More precise within-group score geometry is the immediate bottleneck for a guaranteed bound. Fast adaptive routing needs a cheaper way to estimate that geometry or a separately validated empirical policy.

[Audit code](../../scripts/audit_kam_pretrained_mass_certificate.py) · [Counts by book, layer and query offset](../../results/general_benchmarks_pretrained_sparse_local/certificate.json)
