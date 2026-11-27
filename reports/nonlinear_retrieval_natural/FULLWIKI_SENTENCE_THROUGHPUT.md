# Sentence encoding precision and batched HotpotQA throughput

On 192 warmed L4 requests, moving only shortlisted E5 sentence encoding to FP16 reduced median end-to-end latency from 93.59 to 71.48 ms. FP16 preserved all 192 conditioned pair selections, all 97 complete supports, and all 57 joint exact matches. Independent FP32 batching increased measured throughput by about 22% at eight requests per batch.

## Single-request sentence stage

The fixed search chain uses a full half-precision initial scan with FP32 rescoring, fused 256-list conditioned search, and shortlist-only sentence normalization. A stage replay measured roughly 37 ms in sentence encoding within a roughly 47-ms rerank stage. Sorting each request's sentences by input character length changed the paired median service time by −0.17 ms [−0.44, +0.25]. All pair and reader outcomes were unchanged, and two guard feature records changed. This does not establish a central latency improvement.

Next, the same live questions ran in randomized arm order with resident FP32, FP16, or BF16 E5 models for sentence encoding only. The query encoders, pair scoring, and reader remained fixed. Low-precision embeddings were converted to FP32 before sentence affinity scoring. The search and encoding inventory matched across arms.

| Sentence model | Encoding p50 (ms) | Request p50 (ms) | Request p95 (ms) | Support / joint EM, out of 192 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 37.83 | 93.59 | 143.21 | 97 / 57 |
| FP16 | 14.95 | 71.48 | 103.91 | 97 / 57 |
| BF16 | 14.92 | 71.22 | 104.74 | 96 / 56 |

The median paired request change for FP16 was −23.73 ms [−25.22, −21.45]. FP16 reproduced the whole ranking and reader record on 191 requests. Guard features changed under reduced precision even where pair selection and exactness held. BF16 changed two conditioned pairs, and one of those changes lost support and a joint match. These are matched sample results for the fixed pipeline.

## FP32 throughput

The batch replay timed 24 matched groups of eight requests, in randomized batch-policy order. Each policy used the same corpus and FP32 sentence weights. It shared query and sentence encoder batches and submitted all ready answer prompts in a call. Corpus queries remained separate GPU searches. The timings start with a group already available and exclude time spent waiting for requests to arrive.

| Requests per batch | Requests per second | Median batch service time (ms) | Median paired eight-request speedup |
| ---: | ---: | ---: | ---: |
| 1 | 9.61 | 98.46 | 1.00× |
| 2 | 10.54 | 185.90 | 1.09× |
| 4 | 11.33 | 353.56 | 1.18× |
| 8 | 11.77 | 686.86 | 1.23× [1.20, 1.26] |

Evidence rankings and reader decisions were identical for these 192 requests at all widths. Width eight encoded only 12 fewer sentences than width one, so repetition accounted for little of the gain. The measured throughput benefit accompanies a longer batch service time. Its effect on interactive p95 depends on request arrival and dispatch.

## Confirmation on 800 different requests

An additional 800 requests from the frozen 1,024-question panel excluded every request used in the 192-request precision and batching pilots. The randomized design crossed sentence precision and batch width, with 100 matched groups of eight ready requests. Array 52985779 evaluated all four arms on L4.

| Sentence precision / batch width | Requests per second | Median batch service time (ms) | p95 batch service time (ms) | Support / joint EM, out of 800 |
| --- | ---: | ---: | ---: | ---: |
| FP32 / 1 | 10.61 | 91.86 | 132.39 | 400 / 264 |
| FP16 / 1 | 14.53 | 66.43 | 91.05 | 400 / 264 |
| FP32 / 8 | 12.26 | 648.01 | 773.60 | 400 / 264 |
| FP16 / 8 | 19.83 | 397.60 | 454.95 | 400 / 264 |

The combined policy gave a 1.89× median paired speedup for the eight-request groups [1.86, 1.90]. Each FP16 arm reproduced 797 of 800 ordered conditioned pairs from single-request FP32. Complete support, answer EM, and joint EM agreed on every request. The raw guard values changed, so these results establish parity for the measured pipeline decisions and outcomes. The incremental effects of precision and batching are separately visible in the four arms.

[Timing and exactness records](../../results/nonlinear_retrieval_natural/fullwiki_coarse_v1/) · [First-hop optimization](FULLWIKI_REQUEST_OPTIMIZATION.md)

## Sharing the initial scan

On the same 800 questions, single-GPU batches of eight used one matrix multiply against the resident half-precision corpus, followed by the same FP32 shortlist rescoring per query. A randomized replay compared it with eight separate searches and kept FP16 sentence encoding fixed. Initial retrieval fell from 129.29 to 23.35 ms at the median per eight requests. Median batch completion was 423.38 versus 315.66 ms, with paired median change −109.68 ms [−110.83, −108.16]. Throughput rose from 18.82 to 24.87 requests per second within this run. All 800 complete numerical decision records matched, including both candidate rankings. These results compare two eight-request operating points directly.

HPG array 52987401 ran the L4 shared scan. [Batch and stage measurements](../../results/nonlinear_retrieval_natural/fullwiki_coarse_v1/shared-initial-summary.json)

## B200 replication and reader-precision control

On B200, sharing the initial matrix scan across eight requests reduced median initial search time from 16.18 to 3.48 ms per batch. Median request batch time fell from 310.77 to 299.36 ms, with a paired median change of −18.49 ms (95% batch interval −24.34 to −5.71). Aggregate throughput improved from 25.36 to 26.40 requests/s. All 800 within-device records matched, with 400 complete support sets, 331 exact answers and 264 joint exact answers in both modes. B200 and L4 differed in four full arm records, so within-device optimization parity is the stronger reproducibility claim. The B200 sample batch p95 rose from 376 to 384 ms. [B200 summary](../../results/nonlinear_retrieval_natural/fullwiki_shared_b200_v1/summary.json)

On the same 800 requests on L4, a resident BF16 reader left all selected pairs unchanged. The paired median reader step fell by 10.70 ms (95% batch interval −11.84 to −9.72). Whole-batch throughput rose from 25.77 to 26.28 requests/s, and p95 rose from 358 to 365 ms. BF16 lost five conditioned answers, including four joint successes, with zero gains. The joint change was −0.50 percentage points (95% paired request interval −1.00 to −0.13). Keep FP32 as the quality-preserving reference. [L4 reader summary](../../results/nonlinear_retrieval_natural/fullwiki_reader_precision_v1/summary.json)

HPG arrays 52988596 and 52989122. [Batch analyzer](../../scripts/analyze_kam_batched_initial.py)

## Exact prompt caching and stage profile

Caching the clipped source passages within each eight-request batch reduced passage tokenization from 3,200 calls to 1,775 on L4 and 1,777 on B200. The 100 prompt batch digests and every decision record matched within each device. Paired median prompt preparation fell by 2.68 ms on L4 and 3.29 ms on B200. Whole-batch medians changed by −1.99 ms (95% interval −3.63 to −0.25) on L4 and −6.07 ms (−12.25 to +5.83) on B200. Aggregate throughput was 26.41 versus 26.18 requests/s on L4 and 28.38 versus 28.40 on B200.

A synchronized phase profile attributed about 100 ms per batch to scoring the prepared rerank candidates and 63–76 ms to preparing them. Median FP32 reading was 56–67 ms. The experiment adds phase barriers to each arm, so compare timings within this pair. Exact cache parity is strong, while the end-to-end throughput benefit was limited. [L4 profile](../../results/nonlinear_retrieval_natural/fullwiki_prompt_cache_v1/l4-summary.json) · [B200 profile](../../results/nonlinear_retrieval_natural/fullwiki_prompt_cache_v1/b200-summary.json)

HPG arrays 52990224 and 52990225.
