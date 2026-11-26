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
