# Physical clock compaction on L4 and RTX PRO 6000

Removing a causally inaccessible suffix from a position-free radial decoder reduced depth-1,024 eager FP32 time from 38.72 to 5.35 ms for heterogeneous four-request batches. The same change made balanced BF16 Flash attention slower on this panel and changed two decisions among 192. Physical compaction benefits depend on the baseline kernel and the variation in admitted lengths within a batch.

All arms apply the same clock admission rule and stable partition. Records available by both query clocks come first, then the query, then rejected records. Causal attention cannot access tokens after the query. The full baseline computes over that complete sequence and gathers the state at the query. Batch-maximum cropping removes every token after the longest query position in the batch. Length buckets round each request's eligible span up to a multiple of 64 and issue a separate forward per group. Query positions and admitted records stay fixed. Positions are disabled in the checkpoint. Timing includes the partition, shape selection, synchronization and GPU forward, with warmed weights and input tensors already on-device.

| Depth-1,024 panel | Kernel and batch | Full ms | Cropped to batch maximum ms | Length buckets ms |
| --- | --- | ---: | ---: | ---: |
| Ordinary rejection | Eager FP32, one | 10.60 | 1.52 | 1.63 |
| Ordinary rejection | Eager FP32, four | 38.76 | 9.16 | 9.43 |
| Heterogeneous rejection | Eager FP32, four | 38.72 | 8.96 | 5.35 |
| Heterogeneous rejection | Balanced BF16 Flash, four | 1.51 | 1.56 | 5.71 |

Both crop strategies matched all 2,304 FP32 paired decisions across tested lengths, batches and rejection conditions. The largest observed FP32 logit difference was 7.68×10⁻⁵. The matrix of cases includes two batch shapes per depth, so these counts include repeated inputs. BF16 changes occurred at multiple depths. At depth 1,024 with heterogeneous four-request batches, batch-max crop matched all 192 decisions and length buckets matched 190. A runtime may benefit from choosing different shapes for a slow eager kernel and a fast fused kernel. Shape-specific checks remain important whenever finite precision can change hard decisions.

HPG L4 array 52993078. [Per-seed timings, decision checks and fixture hashes](../../results/nonlinear_retrieval_natural/packed_clock_inference_v1/) · [Clock training and admission](TEMPORAL_CLOCK_ENCODING.md)

## Cross-device replay

On an RTX PRO 6000, the same heterogeneously cropped depth-1,024 eager FP32 batch of four ran at 7.04 ms full, 1.28 ms with batch-max crop, and 3.97 ms with length buckets. Both crop modes preserved all 192 decisions in this panel. Their additional peak allocations were 241.8 and 69.9 MB respectively, down from 1,101.7 MB. Batch-maximum crop was fastest on this GPU, while length bucketing was fastest on L4 for the same panel. The allocation columns count live CUDA allocations above resident inputs and weights.

Balanced BF16 Flash was fastest on the full shape at 1.14 ms on RTX. Batch-max crop took 1.18 ms and buckets took 4.35 ms, each matching 190/192 paired decisions. An inference service that tunes shapes should verify its decisions on the actual GPU and kernel. [HPG RTX PRO 6000 array 53003844](../../results/nonlinear_retrieval_natural/packed_clock_inference_v1/rtx6000-summary.json)
