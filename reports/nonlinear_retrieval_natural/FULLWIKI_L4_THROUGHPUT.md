# FullWiki L4 batch width and integer search

With shared reranker features and cached prompt passages, increasing L4 width from eight to sixteen raised measured throughput from 28.74 to 30.08 requests per second. The next doubling reached 30.38 and increased the peak transient allocation. A separate INT8 first-pass shortlist reduced median paired end-to-end time by 4.32 ms per eight requests at identical evidence and answer outcomes.

## Fixed groups of 32 requests

The width sweep uses 768 frozen HotpotQA bridge requests in 24 groups. All arms use FP16 sentence encoding, FP16 shared first-pass corpus scan with FP32 shortlist rescoring, and the same FP32 reader. The optimized arms additionally share reranker features and cached passage tokens. The crossover randomizes arm order within each group. Timings run from subbatch processing through CUDA-synchronized reader completion. The group timer sums all subbatch times, with requests supplied together and no waiting for arrivals.

| Arm | Requests/s | Median / p95 group ms | Median / p95 subbatch ms | Max extra allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| Reference, width 8 | 26.03 | 1,228 / 1,326 | 307 / 372 | 261 |
| Optimized, width 1 | 14.44 | 2,201 / 2,441 | 67 / 92 | 101 |
| Optimized, width 4 | 24.73 | 1,282 / 1,454 | 159 / 201 | 121 |
| Optimized, width 8 | 28.74 | 1,113 / 1,220 | 278 / 324 | 261 |
| Optimized, width 16 | 30.08 | 1,043 / 1,201 | 525 / 611 | 471 |
| Optimized, width 32 | 30.38 | 1,042 / 1,167 | 1,042 / 1,167 | 644 |

Every arm returned identical ordered evidence pairs and per-question answer and joint flags. All six reached 378 supporting pairs, 315 exact answers, and 250 joint outcomes. Some initial and second-hop rank orders changed with tensor shape. Full records include numerical features and show further differences across widths. The reference at width eight matched all 768 complete records of its optimized counterpart. The paired interval for the change in median 32-request group time at width sixteen relative to optimized width eight was [−73.08, −52.57] ms. The memory column measures maximum allocated CUDA bytes above resident allocations immediately before a group. These are preloaded width measurements, so serving latency for requests that arrive over time also depends on batching and queue delays.

## INT8 first shortlist with FP32 rescore

The isolated replay scans 5.20 million corpus documents on the 1,024 original test requests. Each document has its own INT8 scale. The query is quantized independently, integer products produce an oversampled shortlist, and 96 candidates go through the same ordered FP32 rescore. All 1,024 candidate sets and first titles matched the FP16-scan baseline, with 1,017 exact ordered lists. All 640 complete-pool flags matched. The comparison scanned queries eight at a time. Median isolated scan and rescore was 23.12 ms with FP16 and 20.36 ms with INT8, with a paired difference of −2.74 ms [−2.76, −2.73].

The first-pass score bank decreased from 4,019,196,672 to 2,030,537,472 bytes including INT8 row scales. Both methods retain the separate original FP32 bank for exact shortlist rescoring, and this crossover allocates both first-pass banks. The new bank took 393 ms to build at the median. These figures describe storage for the first pass rather than peak service memory.

In the live crossover, all 800 requests share candidate features, passage caches, frozen weights, and the same FP32 reader. Replacing the first scan preserved the complete ordered second-hop lists, chosen evidence pairs, prompt hashes, and all support, answer, and joint flags. Two of 800 first-hop ranked lists changed within the same set. Both arms reached 400 supporting pairs, 331 exact answers, and 264 joint outcomes. Throughput rose from 29.02 to 29.43 requests per second. Median request time fell from 274.83 to 268.83 ms, with a paired −4.32 ms [−5.14, −1.83]. Median initial search moved from 22.99 to 20.21 ms. Escalating shortlist depth to 192 or 256 preserved the isolated outcomes on the tested fixture, with no recovery benefit over 96.

HPG L4 arrays 52997124, 52995223 and 52997414. [Batch-width records](../../results/nonlinear_retrieval_natural/fullwiki_optimized_widths_v1/) · [Standalone INT8 scan](../../results/nonlinear_retrieval_natural/fullwiki_quantized_search_v1/) · [Live INT8 crossover](../../results/nonlinear_retrieval_natural/fullwiki_live_int8_v1/) · [Feature and prompt parity](FULLWIKI_FEATURE_REUSE.md)

## RTX PRO 6000 scan replay

The same isolated corpus scan ran in 4.27 ms with FP16 and 3.70 ms with INT8, with a paired −0.56 ms [−0.56, −0.56] at two decimal places. Both policies returned the same top title and candidate set on all 1,024 requests and the same ordered list on 1,020. All complete-pool flags matched. This replay used the same 5.20 million documents, eight-query batches, per-document scales and FP32 rescore. The INT8 bank built in 73 ms at the median. The narrow interval covers repeated isolated scan timings within this device and workload. [HPG RTX PRO 6000 array 53003845](../../results/nonlinear_retrieval_natural/fullwiki_quantized_search_v1/rtx6000-summary.json)
