# Article storage and first-touch latency on HPG

The article lookup step took about **32–33 ms at p50** on first touch and **0.6–0.8 ms** on repeated requests after staging. Warm reads from shared Lustre and node-local ext4 were nearly identical. The 1.66 GB node-local copy took **3.57–4.55 seconds**, and both database bytes and all 256 returned article sequences matched exactly.

| CPU shard | Shared first touch p50 / p95 | Shared warmed p50 / p95 | Local warmed p50 / p95 | Copy |
| --- | --- | --- | --- | --- |
| 0 | 32.46 / 49.66 ms | 0.78 / 1.09 ms | 0.80 / 1.19 ms | 4.55 s |
| 1 | 32.83 / 50.62 ms | 0.59 / 0.73 ms | 0.59 / 0.82 ms | 3.57 s |

HPG array **52950517** ran two CPU shards with disjoint samples of 128 questions from the expansion-guard replay. Both read the frozen original and conditioned candidate texts with identical ordered queries. The first pass read shared storage. A full copy to local ext4 followed, then the measured reads of the two stores alternated order. Each read used immutable SQLite with the same mapping settings. We verified filesystem type using the allocation logs and kept decoding outside the timed section.

The schedule exposes a large first-touch penalty and provides a comparison between two warm stores. First-touch requests were not preceded by any deliberate cache eviction. The copy reads the shared source, so the comparison cannot separate repeat access from copy-induced page warming. The results support testing a measured prewarm of the shared index before incurring the storage and deployment cost of a local replica. This benchmark covers article text reads alone.

## Reproduce

Use the [staging driver](../../slurm/nonlinear_retrieval_article_staging.sbatch) with the article index and frozen expansion-guard artifacts. The [replay and audit](../../results/nonlinear_retrieval_natural/article_staging_v1/) record the per-request timings, input hashes, database checksum, and copy duration.
