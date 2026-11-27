# Full-request dense and hybrid timing on HPG B200

At matched initial depth 64, live hybrid search added a **paired median 142.32 ms** over live dense search on 128 HotpotQA questions (95% question-bootstrap interval **122.83–155.59 ms**). Median full-request time was **83.65 ms** for dense and **229.05 ms** for hybrid. Both paths computed the broad-ranked anchor online and used the same conditioned top-64 expansion, sentence reuse, and frozen reader. [Timing audit](../../results/nonlinear_retrieval_natural/live_initial_profile_v1/summary.json)

| Initial search | Full p50 / p95 | Initial search p50 | Anchor p50 | Conditioned search p50 | Remaining rerank p50 | Reader p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dense | 83.65 / 113.85 ms | 2.94 ms | 19.11 ms | 9.40 ms | 19.61 ms | 27.07 ms |
| Exact dense + BM25 RRF | 229.05 / 474.08 ms | 140.26 ms | 19.01 ms | 9.43 ms | 19.56 ms | 27.19 ms |

Hybrid took longer on all 128 timed pairs. The paired median overhead was 128.42 ms when dense ran first and 147.92 ms when hybrid ran first. The initial search phase accounts for most of the difference in this deployment. The hybrid implementation issues GPU scoring before CPU BM25 and fuses the top 128 lists. The article database was prewarmed, while the separate shared lexical files received ordinary warmup queries. These measurements use a serial CPU lexical scorer on the shared index. [Prior threaded lexical results](FULLWIKI_SCALING.md#exact-bm25-with-concurrent-segment-search)

## Reproduction and quality parity

HPG array **52955747** used four B200 shards. Each warmed the shared 1.66 GB article SQLite file and ran four extra warmup requests for both policies. Each measured the two paths in seeded randomized order on 32 questions drawn from the matched 1,024-question quality cohort. The full clock includes online query encoding, first search, article fetch, live anchor selection, conditioned search, both rerank decisions, and the reader. Startup and the per-shard article prewarm of 0.26–0.73 seconds are outside that clock. These are single-request timings. [Driver](../../slurm/nonlinear_retrieval_live_anchor_profile.sbatch)

Each policy reproduced the sealed initial order, selected anchor, both evidence decisions, shortlist labels, and answer exactness on all **128** requests. Expanded top-64 membership matched all 128 for each policy. Expanded order matched 123 for each. The live slice contains 66 dense and 67 hybrid correct support pairs and 45 joint matches per path. In the larger 1,024-question quality comparison, hybrid had 340 joint matches versus 328 for dense, with a paired interval reaching zero. The timing and quality results compare equal depth. They give the cost of the measured implementation at that depth. [Matched quality study](DENSE_HYBRID_ANCHOR.md)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_initial_profile_v1"
python scripts/analyze_kam_live_initial_profile.py "$study"/profile-52955747-*.json.gz \
  --study "$base/dense_hybrid_anchor_v1" --output /tmp/kam-live-initial.json
diff -u "$study/summary.json" /tmp/kam-live-initial.json
```
