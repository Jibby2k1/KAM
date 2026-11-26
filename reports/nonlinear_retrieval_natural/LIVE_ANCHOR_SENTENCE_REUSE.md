# Reusing sentence embeddings within a live anchor request

On 128 paired HPG B200 requests, retaining sentence vectors from the original broad rerank cut the work from **17,603 to 11,781 sentence encodings**, a 33.1% reduction. The paired median request saving was **6.36 ms** (95% question bootstrap **5.70–7.20 ms**). All 128 requests preserved the anchor, ordered corpus hits, selected evidence, shortlist labels, and answer exactness.

| Live broad-ranked policy | Request p50 | Request p95 | Original anchor phase p50 | Rerank after expansion p50 |
| --- | ---: | ---: | ---: | ---: |
| Encode sentences separately | 78.15 ms | 120.96 ms | 16.41 ms | 22.57 ms |
| Reuse vectors within each request | 72.01 ms | 137.76 ms | 16.46 ms | 16.99 ms |

Reuse was faster on 119 of 128 requests. The paired median saving was 7.34 ms when the reference ran first and 5.59 ms when reuse ran first. The pooled sample p95 rose by 16.80 ms. The order-stratified p95 rose in one stratum and fell in the other. An additional concurrency and tail study is needed to characterize serving under load. [Paired timing and parity](../../results/nonlinear_retrieval_natural/live_anchor_sentence_reuse_v1/summary.json)

## What the cache reuses

The first broad rerank determines the anchor before the conditioned corpus search. Its normalized sentence strings identify the inputs to the frozen E5 passage encoder. Within that request, a cache retains those sentence vectors, and the union rerank encodes only the missing strings. Article retrieval, the pair policy, and the reader are shared between timing arms. The cache uses only sentences the model already chose to encode and is cleared at the end of each request. [Live broad anchor](LIVE_ANCHOR_LATENCY.md) · [Runner](../../scripts/run_kam_conditioned_union.py)

The 14 guard features matched exactly as stored on 126 requests. The largest difference was **0.0000167** in a score margin. This measures decision-level parity on the replay. The masked batched encoder can produce small floating-point differences when the group of encoded inputs changes. Both variants matched the archived broad-ranked selections and answer labels on all 128 requests and had the same three adjacent rank swaps relative to the archive. [Archived and paired records](../../results/nonlinear_retrieval_natural/live_anchor_sentence_reuse_v1/)

HPG array **52954950** used four B200 shards. Each prewarmed the 1.66 GB article index, warmed both policy paths on four other questions, and then crossed the policy order on 32 paired requests. The clock begins with frozen initial candidates available and stops after the reader. It includes the original live rerank, conditioned search, expansion rerank, and text fetching. Startup and index readiness are outside the request clock. [Slurm driver](../../slurm/nonlinear_retrieval_live_anchor_profile.sbatch)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/live_anchor_sentence_reuse_v1"
anchor="$base/broad_anchor_v1"
python scripts/analyze_kam_live_anchor_profile.py "$study"/profile-52954950-*.json.gz \
  --fixture "$anchor/fixture.json.gz" --broad "$anchor/replay.json.gz" \
  --sentence-reuse --output /tmp/kam-live-reuse.json
diff -u "$study/summary.json" /tmp/kam-live-reuse.json
```
