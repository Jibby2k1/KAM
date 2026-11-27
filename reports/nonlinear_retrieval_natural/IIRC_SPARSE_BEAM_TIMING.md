# Sparse scoring for the four-link IIRC beam

Scoring only the four-link beam reduced mean pair-feature rows from **101.18 to 47.22**, while preserving every selected pair. On these small pools, indexing overhead outweighed the arithmetic saving at the median on both tested CPUs. The faster sparse implementation added **0.88 μs** on Darwin arm64 and **8.23 μs** on the HPG x86-64 worker at the paired median for link-neutral scores. [Local trace audit](../../results/nonlinear_retrieval_natural/iirc_sparse_beam_v1/apple-summary.json) · [HPG trace audit](../../results/nonlinear_retrieval_natural/iirc_sparse_beam_v1/hpg-summary.json)

| Host and head | Dense p50 | Sparse sorting p50 | Sparse mask p50 | Mask minus dense, paired median |
| --- | ---: | ---: | ---: | ---: |
| Local arm64, neutral | 18.29 μs | 21.02 μs | 19.15 μs | +0.88 μs |
| Local arm64, title-aware | 18.38 μs | 21.00 μs | 19.17 μs | +0.81 μs |
| HPG x86-64, neutral | 76.09 μs | 96.16 μs | 84.23 μs | +8.23 μs |
| HPG x86-64, title-aware | 76.79 μs | 96.45 μs | 84.63 μs | +7.99 μs |

Each method assembled the chosen symmetric pair features, applied the same fixed conditional scores, normalized each first-link branch, combined unordered-pair mass, and returned the same highest-scoring pair. Dense assembled the feature table for all candidate pairs before gathering the branches. The two sparse implementations deduplicated branch edges before assembling features, using either a sort or a Boolean mask. The benchmark verified each result against its earlier out-of-fold prediction. [Anchor beam definitions](IIRC_ANCHOR_BEAM.md) · [Benchmark implementation](../../scripts/benchmark_kam_iirc_sparse_beam.py)

The test used all 1,200 questions and the same source-fold fits. Each worker ran eight repeats per question, head, and implementation in seeded randomized order. The summary compares each question's median across repeats. HPG job **52959948** used one allocated CPU core. The hosts use different Python and NumPy versions, so report contrasts within each host. The clock starts from resident numeric candidate and relation columns. It excludes document encoders, offline relation extraction, and answer decoding. A sparse kernel that also avoids expensive pair-dependent feature extraction may have different scaling. [HPG launcher](../../slurm/nonlinear_retrieval_iirc_sparse_beam.sbatch)

## Reproduce the trace audit

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_sparse_beam_v1"
python scripts/analyze_kam_iirc_sparse_beam.py --profile "$study/hpg-profile.json.gz" \
  --reference "$base/iirc_anchor_beam_v1/replay.json.gz" \
  --static "$base/iirc_pair_fit_v1/train-features.json.gz" --output /tmp/kam-iirc-sparse.json
diff -u "$study/hpg-summary.json" /tmp/kam-iirc-sparse.json
```
