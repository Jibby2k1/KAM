# KAM research results

## Multi-hop evidence selection

The [held-out HotpotQA supporting-link study](../../reports/nonlinear_retrieval_natural/SUPPORT_LINKS.md) compares a question-conditioned pair scorer with one-pass and sequential retrieval on shared dense pools. It reports linked and unlinked gold evidence separately and tests the ranking of supporting links against distractor links directly.

The [coverage-gate study](../../reports/nonlinear_retrieval_natural/COVERAGE_GATE.md) tests a learned choice between two pair heads using observable candidate features. It measures clean, sham, and false-link evidence recovery on 720 additional questions.

A [sentence reranker](../../reports/nonlinear_retrieval_natural/EVIDENCE_VERIFIER.md) recovered both supports on 252/420 2Wiki and 88/300 MuSiQue questions with standard false links, up from 122/420 and 61/300 for the frozen context head. On native Wikipedia HotpotQA pools, joint exact match fell from 231 to 217 of 768.

In a [fresh HotpotQA refit study](../../reports/nonlinear_retrieval_natural/NATIVE_SENTENCE_REFIT.md), native sentence training recovered 392/450 naturally linked support pairs, compared with 363 for the transferred 2Wiki policy and 388 for the frozen scorer. A native fit on the existing pair features recovered 387.

On [fresh frozen Wikipedia pools](../../reports/nonlinear_retrieval_natural/NATIVE_ROUTING_GATE.md), the inference-visible gate recovered both supports on 486/1,024 and joint exact answers on 275/1,024. Always using the broad native reranker reached 487 and 276. The two-reranker oracle reached 492 and 280.

On [the same sealed pools](../../reports/nonlinear_retrieval_natural/FROZEN_SEQUENTIAL_ORACLE.md), learned sequential retrieval reached 174 support recoveries and 99 joint exact matches. An oracle choosing between sequential and broad reached 506 and 291, with most of the extra successes on unlinked questions.

The [conservative sequential fallback](../../reports/nonlinear_retrieval_natural/SEQUENTIAL_FALLBACK.md) stayed with broad on all 1,024 fresh pools after calibration, yielding 452 complete support recoveries and 285 joint exact matches. The 5% calibration threshold yielded 450 and 283. Signals detected unlinked gold evidence better than they detected the cases where switching helped.

[Conditioned full-corpus second-hop search](../../reports/nonlinear_retrieval_natural/CONDITIONED_CORPUS_SEARCH.md) recovered both supports on 11 incomplete pools and reached six joint exact matches. Across all 1,024 questions it reached 173 support recoveries and 110 joint exact matches, compared with 178 and 116 from same-pool sequential and 452 and 285 from broad.

[Reranking the conditioned union](../../reports/nonlinear_retrieval_natural/CONDITIONED_UNION.md) raised support recovery from 481 to 511 and joint exact match from 308 to 333 on 1,024 fresh HotpotQA pools. The union recovered 46 support pairs and 35 joint answers from 384 incomplete pools.

[Three-cohort replication](../../reports/nonlinear_retrieval_natural/EXPANSION_REPLICATIONS.md) found 104 net joint exact matches from affinity-anchored expansion on 3,072 disjoint HotpotQA bridge questions. The union gained 126 joint matches on incomplete pools and lost 22 net on complete pools. It completed none of the 108 pools missing both supports.

[Fixed-budget multi-anchor expansion](../../reports/nonlinear_retrieval_natural/MULTI_ANCHOR_EXPANSION.md) completed 105 of 407 incomplete pools with one anchor, 109 with two, and 113 with four. Alternate anchors improved coverage when the first was wrong and reduced it when the first was correct.

The [expansion replacement guard](../../reports/nonlinear_retrieval_natural/EXPANSION_REPLACEMENT_GUARD.md) preserved more complete-pool support pairs but gave up 16 joint matches overall versus always using the conditioned union. The [exact bitset benchmark](../../reports/nonlinear_retrieval_natural/UNION_RELATION_FEATURES.md) matched all 2,048 feature inputs and accelerated the CPU feature step by about 1.3×.

[Shared relation extraction](../../reports/nonlinear_retrieval_natural/SHARED_RELATION_FEATURES.md) reproduced all evidence selections and reader labels while reducing median paired feature extraction from 12.68 to 7.51 ms.

[Broad-ranked anchoring](../../reports/nonlinear_retrieval_natural/BROAD_RANKED_ANCHOR.md) raised the conditioned union from 356 to 369 joint exact matches on 1,024 new questions. Twelve of the 13 net gains came from initially incomplete pools.

[Primed request timing](../../reports/nonlinear_retrieval_natural/SHARED_SERVING.md) found a 4.37 ms paired median saving for shared relation extraction and a higher empirical p95. [Exact article staging](../../reports/nonlinear_retrieval_natural/ARTICLE_STAGING.md) found similar lookup times for warmed shared and node-local stores.

[Shared-index prewarming](../../reports/nonlinear_retrieval_natural/ARTICLE_PREWARM.md) reduced median article lookup from 34.73 to 0.54 ms across 256 exact crossed requests on two HPG nodes.

The [gold-evidence reader diagnostic](../../reports/nonlinear_retrieval_natural/GOLD_READER_ORACLE.md) found 611 exact answers with indexed gold articles and 660 with annotated supporting sentences, a paired 4.79 percentage point gain on 1,023 valid questions.

[Automatic sentence selection](../../reports/nonlinear_retrieval_natural/READER_SENTENCE_SELECTION.md) tied the full reader prefix at 369 joint exact matches with a companion-title selector. The fixed lead-two baseline reached 336, and query-only selection reached 367.

[Document-order replay](../../reports/nonlinear_retrieval_natural/READER_DOCUMENT_ORDER.md) changed 76 joint outcomes on the same fixed pairs. Reversal gained 40 and lost 36, with an aggregate interval spanning zero.

The [dataset breadth handoff](../../reports/nonlinear_retrieval_natural/DATASET_BREADTH_2026-11-26.md) includes independent Wikipedia, news and scholarly evidence, real temporal revisions, noisy conversation retrieval, and a position and length study of the KAM decoder. See the [dataset portfolio](DATASET_PORTFOLIO.md) for source and evaluation choices.

## Corpus retrieval and serving

At matched isolated latency on two independent HotpotQA cohorts (512 questions total), exact dense retrieval put both gold documents in the candidate pool for 349 questions, compared with 302 for hybrid and 266 for BM25. Joint exact match was 85, 89, and 91 respectively. Paired intervals covered zero. The change in rank from pool coverage to downstream joint exact match identifies pair selection as a separate bottleneck. [L4 aggregate](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/exact_threads_l4/summary.json) · [Independent replication](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/exact_replicate_l4/summary.json)

On an RTX GPU, FP32 lexical pruning in the hybrid pipeline reduced p95 latency from 240–253 ms to 78–80 ms across two traces offered at about 32 requests per second. Joint exact match was 53 versus 52 for the exact pipeline on the 384-question replication. Fifteen answers changed and the paired interval covered zero. [Replication results](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/hybrid_replicate_v1/summary.json)

## Revision memory

A frozen conservative gate rejected all 576 prompts for 96 unseen registry identities while answering 48 of 192 supported-alias prompts. Supplying a verified identity raised supported coverage to 192 of 192. These figures measure coverage and abstention on that fixture. [Registry holdout results](../../results/nonlinear_retrieval_natural/release_registry_sample_v1/summary.json)

The [archived revision study](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#real-revision-and-abstention) measures the distinct effects of identity, observation date, publication cutoff, and simulated local ingestion delays.

The [independent Dallas Fed replication](../../reports/nonlinear_retrieval_natural/OECD_REVISIONS.md) tests typed as-of lookup against an exact oracle. On observed 48-candidate pools, radial recovery rose from 3.8% to 57.6% after procedural training incorporated clusters of real release patterns.

The [matched revision and abstention follow-up](../../reports/nonlinear_retrieval_natural/REVISION_CONTROLS.md) holds records and positions fixed. Hiding extra same-address releases cost dot 13.1 points and radial 1.4 points. Hiding equally many distractors improved both. On unavailable historical cutoffs, confidence thresholds transferred poorly from the Philadelphia Fed to the Dallas Fed archive.

## Model and controlled experiments

See the [architecture specification](../../Kernel_Adaptive_Memory_Research_Specification.pdf), the [core experiment guide](../../EXPERIMENTS.md), and the [Phase IV–VI documentation index](../../LLM_NAVIGATION.md) for controlled KAM model experiments. Natural-task experiments evaluate the transfer of routing ideas using separate retrieval components.

The [controlled revision-noise report](../../reports/nonlinear_retrieval_natural/BITEMPORAL_MODEL.md) tests exact temporal pointers, absolute position, radial versus dot attention, token-matched training curricula, and exact batched pair reduction on two GPU families.

[Position-preserving context compaction](../../reports/nonlinear_retrieval_natural/CONTEXT_COMPACTION.md) preserved all decisions in a numerical probe and shortened the batched radial forward by about 5.3× on L4 and 2.5× on RTX when half of 257 slots were active.

The [B200 serving prewarm crossover](../../reports/nonlinear_retrieval_natural/SHARED_PREWARM_SERVING.md) reduced median affinity-anchor request time from 132.34 to 63.29 ms and p95 from 203.50 to 91.50 ms. All 256 crossed requests retained identical evidence and answer exactness.

At equal depth 64 and with the same B200 broad-ranked expansion, [hybrid initial search](../../reports/nonlinear_retrieval_natural/DENSE_HYBRID_ANCHOR.md) increased initial coverage from 603 to 659 of 1,024 questions. After expansion, it produced 543 support pairs and 340 joint exact matches versus 526 and 328 for dense. The paired joint EM interval reached zero.

The [live B200 anchor profile](../../reports/nonlinear_retrieval_natural/LIVE_ANCHOR_LATENCY.md) measured an 11.29 ms paired median overhead for broad-ranked anchoring versus affinity across 128 prewarmed requests. Both policies reproduced their archived anchors, candidate membership, evidence selections, and exactness.

On 153 [IIRC questions with two linked gold sources](../../reports/nonlinear_retrieval_natural/IIRC_PAIR_TRANSFER.md), sequential recovered 40 pairs, independent top-two 32, and frozen pair scoring 22. Masking between-candidate title mentions recovered 30. Title features helped the small between-gold mention slice and hurt the larger remaining slice.

[Within-request sentence reuse](../../reports/nonlinear_retrieval_natural/LIVE_ANCHOR_SENTENCE_REUSE.md) removed 33.1% of sentence encodings and saved 6.36 ms at the paired median on B200. All 128 requests preserved retrieval and reader decisions. The sample p95 rose from 120.96 to 137.76 ms.

With gold IIRC article identities supplied, a [matched text-window audit](../../reports/nonlinear_retrieval_natural/IIRC_EVIDENCE_VISIBILITY.md) exposed all linked spans on 98 of 139 fully aligned questions with the first 160 words, on 9 with question-only BM25, and on 134 with annotation-informed windows at the same budget.

A [HotpotQA expansion stage audit](../../reports/nonlinear_retrieval_natural/EXPANSION_STAGE_HEADROOM.md) found 244 unresolved coverage misses, 237 ranking misses despite coverage, and 203 inexact answers after recovering both supports. An oracle over the four existing B200 policies reaches 361 joint matches versus 340 for hybrid expansion.

On 134 aligned IIRC cases with the two gold linked articles fixed, the [frozen reader](../../reports/nonlinear_retrieval_natural/IIRC_FIXED_ARTICLE_READER.md) reached 25 exact answers with article leads, 26 with oracle linked windows, and 38 with annotated source and linked spans. Most gains were on single-span answers.

Training on separate IIRC articles raised pair recovery to 56/153 with a [hyperlink-context independent ranker](../../reports/nonlinear_retrieval_natural/IIRC_IN_DOMAIN_PAIR.md), 66 with a link-neutral pair head, and 73 with title features. The matched primary pair gain over independent ranking was 6.54 points, with a paired interval reaching zero.

On the frozen hybrid HotpotQA unions, [widening the shortlist](../../reports/nonlinear_retrieval_natural/HOTPOT_SHORTLIST_DEPTH.md) from 16 to 64 proposals per head admitted 42 more gold pairs and added three joint exact matches. The mean distinct sentence count rose from 72.78 to 185.84.

With the same gold articles and frozen reader, a [question-derived IIRC answer format](../../reports/nonlinear_retrieval_natural/IIRC_READER_FORMAT.md) increased exact answers from 25 to 42 of 134. Binary exactness rose from 1/25 to 17/25, and numeric exactness reached 1/19.

With the same supervision and fixed IIRC representations, the [matched neutral joint and conditional heads](../../reports/nonlinear_retrieval_natural/IIRC_MATCHED_SEQUENTIAL.md) recovered 66 and 62 of 153 pairs. Title-aware joint and conditional heads recovered 73 and 65. The first hop selected a gold article on 91 questions.

The [live first-search benchmark](../../reports/nonlinear_retrieval_natural/LIVE_DENSE_HYBRID_TIMING.md) timed the complete expanded request on B200 at equal depth 64. Dense and hybrid medians were 83.65 and 229.05 ms, with a +142.32 ms paired hybrid overhead. All 128 archived decisions matched per arm.

With IIRC first hops fixed, [second-hop query re-encoding](../../reports/nonlinear_retrieval_natural/IIRC_DYNAMIC_SECOND_HOP.md) left neutral pair recovery at 62/153 and moved title-aware recovery from 65 to 63. The extra encoding pass gave no exact-recovery gain on this cohort.

[Exact lexical prewarming and threading](../../reports/nonlinear_retrieval_natural/LEXICAL_PREWARM.md) reduced CPU search p50 from 150.64 to 58.22 ms, preserving all ordered candidates. The [B200 collection trace](../../reports/nonlinear_retrieval_natural/LIVE_SENTENCE_REUSE_GC.md) found 83–113 ms generation-2 pauses on all four recurring sentence-reuse tail spikes.

With prewarmed files, [threaded B200 hybrid serving](../../reports/nonlinear_retrieval_natural/LIVE_WARM_LEXICAL.md) saved 4.64 ms paired median versus serial hybrid with identical decisions. The threaded hybrid path still added 38.12 ms paired median over dense at equal depth.

In source-held-out IIRC folds, a [four-link anchor beam](../../reports/nonlinear_retrieval_natural/IIRC_ANCHOR_BEAM.md) recovered 563/1,200 neutral and 598/1,200 title-aware pairs, versus 539 and 547 with one link. The four-link subset contained about 47% as many distinct pairs as exhaustive joint scoring.

[Exact sparse IIRC beam scoring](../../reports/nonlinear_retrieval_natural/IIRC_SPARSE_BEAM_TIMING.md) reduced mean scored pair rows from 101.18 to 47.22 but increased median latency on both tested CPUs. All four-link selections matched.

A [B200 collector readiness trial](../../reports/nonlinear_retrieval_natural/LIVE_GC_READINESS.md) moved four observed generation-2 pauses out of the short request window. Readiness collection took 87–142 ms per worker. Additional freezing showed no clear paired median improvement over collection alone.

[Radial attention backward checks](../../reports/nonlinear_retrieval_natural/RADIAL_BACKWARD.md) met FP32 numerical tolerances on both HPG devices. At 1,025 tokens and batch 32, BF16 FlashAttention accelerated the full step by 3.08× on B200 and 18.99× on L4. Its depth-512 accuracy after a matched continuation rose 3.32 points on B200 and fell 4.69 on L4.

A [duration-only serial schedule audit](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#executable-labels-for-fictional-event-schedules) reproduced the published answer sets for 2,693 of 2,700 hard UnSeenTimeQA records. The remaining seven had repeated event descriptions in hypothetical queries.

[BF16 radial feature precision](../../reports/nonlinear_retrieval_natural/RADIAL_BF16_PRECISION.md) reduced gradient relative error at depth 512 across three seeds on both GPUs. The combined encoding retained 3.08× B200 and 19.11× L4 speedups over eager BF16. Relative to original Flash, its mean depth-512 accuracy changed −0.26 points on B200 and +1.50 on L4.

In the [sustained B200 collector replay](../../reports/nonlinear_retrieval_natural/LIVE_GC_SUSTAINED.md), readiness collection shifted all four observed full pauses out of 1,008 cached requests per policy. All policy outcomes matched. Pooled p99 went from 159.91 to 153.75 ms. Adding freezing did not improve p99 consistently across workers.

The [parallel schedule audit](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#parallel-schedules) matched 2,618 of 2,700 released hard parallel answers from text-derived dependencies. Using the authors' dependency schedules, 2,675 had a unique active-event answer, 15 had competing active events, and ten had ambiguous timing interventions.

On [1,173 matched serial duration swaps](../../reports/nonlinear_retrieval_natural/TEMPORAL_DATASETS.md#duration-changes-with-the-same-numbers), none of the tested frozen semantic policies selected both exact events before and after the label-changing edit. The best clean selector reached 330 top-five hits, compared with 275 after the duration swap.

An [exact serial duration index](../../reports/nonlinear_retrieval_natural/SERIAL_DURATION_INDEX.md) matched 3,519 authored-domain selections and 5,760 boundary queries on both CPUs. At 16,384 events, HPG lookup took 1.15 μs versus 446.97 μs to rebuild the timeline, with 403.15 μs to precompute the reusable prefix.

With a typed anchor and elapsed time, [learned dot and radial event pointers](../../reports/nonlinear_retrieval_natural/DURATION_POINTER.md) recovered both sides of 52.77% and 49.53% of authored schedule edits. The local pointer recovered 1.08%. Both context pointers fell below the mean-duration rule at held-out lengths of 64 and 128.

In [matched from-scratch radial precision training](../../reports/nonlinear_retrieval_natural/RADIAL_ARCHIVE_TRAINING.md), the combined BF16 Flash encoding stayed within 0.47 points of eager FP32 on the earlier 48-record archive cohort and within 2.08 on the later one. Eager FP32 finished the full short-sequence training schedule fastest on both GPUs.

Coupled [clock and position scaling](../../reports/nonlinear_retrieval_natural/DURATION_COORDINATES.md) raised exact paired duration-swap recovery from 51.27% to 62.92%. The wide normalized curriculum recovered 23.63% at 256 events, compared with 20.12% for a suffix-mean control.

A [cross-dataset index audit](../../reports/nonlinear_retrieval_natural/CROSS_DATASET_WIKIPEDIA.md) found every supporting title for 10,096/12,576 2Wiki and 2,046/2,417 MuSiQue questions. Complete exact annotation coverage was 2,488 and 150 respectively, with sentence and paragraph units as released by each dataset.
