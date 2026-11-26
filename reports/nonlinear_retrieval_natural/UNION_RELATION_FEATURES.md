# Exact bitset relation features for HotpotQA unions

Bitsets and integer mention counts reduced median CPU feature-extraction time by **1.30–1.33×** on the 1,024-question conditioned-union cohort. The title graph and all five lexical relation features matched bitwise across **2,048 inputs**, covering each original pool and its conditioned union.

## Measurement

The baseline uses Python sets to count shared or covered query terms and scalar tensor writes for the local mention matrix. The comparison packs each document's query terms into a Python integer, computes cardinalities with bit operations, and keeps integer mention counts until conversion to the original floating-point format. The title trie uses the same full-title matches and mention windows as the baseline. Order alternated by question and each comparison used identical indexed articles and paired candidate indices.

| HPG CPU shard | Original p50 / p95, baseline → bitset | Union p50 / p95, baseline → bitset |
| --- | --- | --- |
| 0 | 6.46 / 8.84 → 4.92 / 5.84 ms | 15.05 / 19.11 → 11.46 / 13.74 ms |
| 1 | 7.42 / 10.26 → 5.71 / 6.72 ms | 17.29 / 22.70 → 13.02 / 15.70 ms |

Each AMD EPYC 7702 shard evaluated 512 distinct questions using one CPU thread with PyTorch 2.8.0. HPG array **52949303** produced both profiles. A prior prototype failed exact title-edge parity because it restarted the trie for each token. The corrected traversal passed 36 generated edge cases and all 2,048 real inputs. Timing excludes article reads, corpus search, GPU ranking, and generation. The benchmark measures the feature function in isolation.

## Reproduce

Use the [benchmark and Slurm driver](../../slurm/nonlinear_retrieval_union_relation_features.sbatch) with the existing SQLite article index, conditioned union fixture, and archived expansion replay. The [timing artifacts](../../results/nonlinear_retrieval_natural/union_relation_features_v1/) include both raw profile shards. Recompute summaries and verify cohort membership and input hashes with the analyzer.

```bash
base=results/nonlinear_retrieval_natural
union="$base/conditioned_union_v1"
timing="$base/union_relation_features_v1"
python scripts/analyze_kam_union_relation_features.py "$timing"/profile-?.json \
  --fixture "$union/fixture.json.gz" --pools "$union/pools.json.gz" --replay "$union/replay.json.gz" \
  --output /tmp/kam-union-features.json
diff -u "$timing/summary.json" /tmp/kam-union-features.json
```
