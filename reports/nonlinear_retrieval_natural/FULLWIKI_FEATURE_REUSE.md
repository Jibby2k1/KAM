# Exact reranker feature reuse and batched second-hop search

On the 800-question HotpotQA serving panel, exact feature reuse improved L4 throughput from 25.27 to 27.67 requests per second. The optimization matched every saved reranker feature, prompt, ranked candidate and decision in the paired run. Batching the conditioned lookup also matched all decisions, with little impact on batch time.

Both replays use the same 100 batches of eight recent held-out requests and the same shared first-hop scan. The sentence encoder is resident in FP16, the reader is FP32, and the conditioned lookup probes 256 lists. The feature optimization vectorizes shallow relation bit counts across document pairs and shares each document's question overlap and best sentence across the candidate pairs. The reference computes the same information per pair. For questions with more than 64 distinct eligible terms, bit counting uses the arbitrary-precision reference. A separate local check matched 16 relation matrices and 1,092 sentence-pair feature vectors across empty queries, boundary widths, ties and title mentions.

| L4 comparison | Timed stage | Reference p50 | Optimized p50 | Paired median change, with 95% interval |
| --- | --- | ---: | ---: | ---: |
| Feature reuse | Rerank preparation | 63.61 ms | 41.31 ms | −22.47 ms [−23.12, −22.00] |
| Feature reuse | Request batch | 310.30 ms | 284.78 ms | −24.53 ms [−25.43, −23.35] |
| Batched second hop | Conditioned encoding and search | 30.45 ms | 30.52 ms | +0.13 ms [+0.05, +0.23] |
| Batched second hop | Request batch | 304.18 ms | 305.12 ms | −0.15 ms [−1.43, +1.35] |

In the feature study, batch p95 fell from 363.51 to 336.47 ms. All 100 digests of the rerank matrices after existing rounding and all 100 exact reader prompt digests matched. Each second-hop arm keeps the same individual centroid selection and FP32 shortlist rescoring, and the batched arm uses one shared Triton posting-score launch. All 800 ordered second-hop lists matched. The timed phases use the same CUDA synchronization points within each experiment. Each reports an independently repeated reference rather than combining timing effects across different allocations.

HPG arrays 52992772 and 52993386. [Feature reuse results](../../results/nonlinear_retrieval_natural/fullwiki_feature_reuse_v1/) · [Second-hop batching results](../../results/nonlinear_retrieval_natural/fullwiki_second_batch_v1/) · [Shared initial corpus scans](FULLWIKI_SENTENCE_THROUGHPUT.md#sharing-the-initial-scan)
