# Coarse-to-fine conditioned Wikipedia search

On NVIDIA L4, probing 256 of 2,048 spherical posting lists with a fused scoring kernel reduced conditioned corpus search median from 32.61 to 2.63 ms. On 1,024 frozen HotpotQA questions the resulting candidate ranks changed one joint exact match, from 333 to 332. A full FP16 scan followed by FP32 rescoring preserved all selected pairs and exactness labels and reduced live median request time by 16.30 ms. The fused 256-list path reduced it by 30.01 ms.

## Search replay

A fixed sample of 65,536 eligible Wikipedia vectors trained 2,048 spherical centroids for 12 iterations. All 5,201,800 eligible article rows were then assigned to a centroid. No labeled questions entered partition training. Each query probes the closest centroids, scores their FP16 passage vectors with FP32 accumulation, and reranks the best 96 vectors using their original FP32 values. It returns 64 additions to the frozen initial pool. All methods used the same adapted conditioned query and excluded the same conditioning article. The evaluation checked that exact search reproduced all 1,024 archived candidate sets. [Frozen corpus-expansion experiment](CONDITIONED_UNION.md) · [Inverted-list indexing background](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

| L4 search policy | Median scanned rows | Search p50 | Search p95 | Complete originally incomplete pools, 384 |
| --- | ---: | ---: | ---: | ---: |
| All rows, FP32 | 5,233,329 | 32.61 ms | 32.75 ms | 77 |
| FP16 full scan and FP32 rescore | 5,233,329 | 16.13 ms | 16.19 ms | 77 |
| 16 posting lists | 44,395 | 0.53 ms | 0.74 ms | 58 |
| 64 posting lists | 166,390 | 2.00 ms | 2.36 ms | 70 |
| 256 posting lists | 660,452 | 7.24 ms | 7.87 ms | 76 |
| 64 posting lists, fused | Same as 64 | 0.96 ms | 1.08 ms | 70 |
| 256 posting lists, fused | Same as 256 | 2.63 ms | 2.81 ms | 76 |

The fused Triton kernel reads vector tiles directly from the packed posting lists. It preserved the complete ordered top 64 for all 1,024 queries at each tested probe width relative to the gathered-vector implementation. In the same allocation as its gathered counterpart, the fused 256-list median was 2.63 ms versus 7.22 ms and the fused 64-list median was 0.96 ms versus 1.99 ms. Offline search timing includes the list selection, FP32 rescoring, and transfer of the ordered ranks. The benchmark had the full and packed FP16 banks and the FP32 vectors resident together. [Triton API](https://triton-lang.org/main/python-api/triton.language.html)

## B200 search-only replay

Using the same partition, candidate membership matched between L4 and B200 for every query and every gathered search policy. Exact ordered ranks matched on 1,021 of 1,024 queries across devices. Full scan plus rescoring matched exact candidate membership on all B200 questions, with three ordering differences within the B200 run.

| B200 search policy | Search p50 | Search p95 | Complete originally incomplete pools, 384 |
| --- | ---: | ---: | ---: |
| All rows, FP32 | 2.98 ms | 2.99 ms | 77 |
| FP16 full scan and FP32 rescore | 1.57 ms | 1.59 ms | 77 |
| 16 posting lists | 0.37 ms | 0.63 ms | 58 |
| 64 posting lists | 0.57 ms | 0.64 ms | 70 |
| 256 posting lists | 1.64 ms | 1.84 ms | 76 |

The full-scan rescore was slightly faster than gathered probe 256 on B200 and kept the extra complete pool. This reverses their search-only latency ordering on L4. B200 results in this table are for the gathered implementation and resident index. HPG array 52976442 ran the B200 replay. The downstream request profiles below used L4.

The B200 fused replay used the same frozen ranks and partition. At 256 probes, fusion lowered median search time from 1.74 to 0.68 ms within the randomized replay. At 64 probes, it moved from 0.60 to 0.50 ms. Both fused policies matched their gathered ordered results on all 1,024 requests. Array 52986550 ran the matched fused comparison.

## Fixed downstream quality

| Conditioned additions | Support / joint EM, all 1,024 | Originally complete, 640 | Originally incomplete, 384 |
| --- | ---: | ---: | ---: |
| Exact FP32 | 511 / 333 | 465 / 298 | 46 / 35 |
| Full scan plus FP32 rescore | 511 / 333 | 465 / 298 | 46 / 35 |
| Probe 64 | 508 / 330 | 466 / 298 | 42 / 32 |
| Probe 256 | 510 / 332 | 465 / 298 | 45 / 34 |

The all-row rescore preserved every candidate set, with two order changes, and preserved all final ordered pairs and measurements. Probe 256 matched 98.17% of exact candidates on average and the final ordered pair on 1,020 questions. Its paired joint change was −0.10 points (95% interval −0.29 to 0.00). Probe 64 matched 91.37% of candidates on average and changed joint EM by −0.29 points (−0.78 to +0.10). The same reranker and reader were used for every sealed replay. All archived broad and unconditioned controls reproduced exactly.

## Randomized live requests

The four L4 workers each timed 48 requests in randomized arm order after prewarming the article store and model shapes. Every arm retrieved the initial dense pool live and used the same initial pool and anchor on each request. Timing includes encoding both queries, initial and conditioned corpus search, reading article text, reranking both the broad and expanded pools, and the frozen reader. The query encoder and reader are outside the search-only table above.

| Live conditioned search arm, 192 requests | Request p50 / p95 | Conditioned stage p50 | Median paired change from exact |
| --- | ---: | ---: | ---: |
| Exact FP32 | 142.70 / 195.68 ms | 35.16 ms | 0.00 ms |
| Full scan plus FP32 rescore | 126.56 / 179.48 ms | 18.70 ms | −16.30 ms |
| Gathered probe 256 | 117.63 / 170.29 ms | 10.00 ms | −25.48 ms |
| Fused probe 256 | 112.55 / 165.07 ms | 5.24 ms | −30.01 ms |
| Fused probe 64 | 110.93 / 160.68 ms | 3.57 ms | −31.87 ms |

Every arm selected both supports on 97 requests and attained 57 joint exact matches. The full rescore preserved every selected pair on the live sample. The probe variants each matched 191 of 192 pairs. Their small live-sample outcome change and larger frozen-split comparison should be considered together. Conditions use the same hardware and warm cache regime.

HPG L4 jobs 52975618 and 52976441 built and queried the partition. Array 52978577 timed the fused scoring kernel. Reranking jobs 52977498–52977500 replayed the frozen candidates. Array 52979797 ran the live comparison. [Search and request measurements](../../results/nonlinear_retrieval_natural/fullwiki_coarse_v1/) · [Build and search](../../scripts/run_kam_fullwiki_coarse.py) · [Fused score kernel](../../scripts/kam_coarse_scores.py)
