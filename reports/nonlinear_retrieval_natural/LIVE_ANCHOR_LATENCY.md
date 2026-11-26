# Live broad-ranked anchor latency on B200

Computing the broad-ranked anchor during serving added a **paired median 11.29 ms** over affinity anchoring (95% question bootstrap **10.61–12.49 ms**). End-to-end request p50 rose from **69.29 to 81.81 ms**, and p95 from **108.71 to 146.88 ms**, after the original top-64 pool was available.

| Policy | Request p50 / p95 | Anchor p50 | Conditioned query and search p50 | Remaining rerank p50 | Reader p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affinity | 69.29 / 108.71 ms | 0.16 ms | 9.03 ms | 30.45 ms | 24.92 ms |
| Live broad-ranked | 81.81 / 146.88 ms | 18.22 ms | 8.86 ms | 24.13 ms | 25.53 ms |

The broad-ranked route computes the frozen broad reranker on the original pool, uses its first predicted article for expansion, and reuses its original result for guard features and answer comparison. It builds exact bitset relation features before and after the conditioned search. The affinity route can defer both rerankers until after expansion and share the relation extraction and sentence encoding pass. The phase medians describe individual stages and do not sum to the total median. The total also includes initial candidate text fetch and question encoding. [Audited timing and traces](../../results/nonlinear_retrieval_natural/live_anchor_profile_v1/)

## Comparison and fidelity

HPG array **52953830** used B200 for all four shards. Each prewarmed the 1.66 GB article database once (0.24–0.74 seconds), ran both paths on four warmup questions outside the sample, and measured both paths in seeded randomized order on 32 questions. The paired median overhead was 10.59 ms when affinity ran first and 13.64 ms when live broad ran first. These are single-request measurements with the frozen initial pools and warm models. They exclude the initial corpus search, process startup, and readiness prewarm. [Driver](../../slurm/nonlinear_retrieval_live_anchor_profile.sbatch)

For both policies, all 128 anchors, candidate sets, selected pairs, shortlist labels, and reader exactness matched their respective batched quality replays. The ordered corpus list matched on 127 affinity and 125 broad-ranked questions. The remaining lists differed by one adjacent swap each, with all 64 members retained. The original broad scorer produced identical outputs under the two live routes for all 128 questions. In the full 1,024-question quality replay, broad-ranked anchoring gained 19 support pairs and 13 joint matches versus affinity. This timing audit attaches a measured latency cost to that comparison on the archived dense cohort. Hybrid initial-search latency needs a separate measurement. [Quality replay](BROAD_RANKED_ANCHOR.md)

## Reproduce the audit

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_anchor_profile_v1"
anchor="$base/broad_anchor_v1"
python scripts/analyze_kam_live_anchor_profile.py "$study"/profile-52953830-*.json.gz \
  --fixture "$anchor/fixture.json.gz" --affinity "$anchor/affinity-replay.json.gz" \
  --broad "$anchor/replay.json.gz" --output /tmp/kam-live-anchor.json
diff -u "$study/summary.json" /tmp/kam-live-anchor.json
```
