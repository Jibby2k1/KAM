# Overnight routing and source selection

The prior MuSiQue study composed the two sequential decisions from separately stored model outputs. This follow-up executes both decisions online for each request. It compares direct choice, staged choice and mirrored endpoint scoring with the three matched Llama 3 KAM and dot checkpoints at KL 0.5, plus the untouched reference. [Prior MuSiQue results](PRETRAINED_ENDPOINT_CHOICE.md)

## Frozen comparisons

- **MuSiQue routing.** Exclude the IDs, normalized questions, relation families and supporting titles from the prior endpoint panel. Match two-hop facts within each source split and relation family. Report the publisher's development and training strata separately. The training-stratum label refers to the source split, and the Llama adapters are unchanged. Require correctness across all four earlier-passage and endpoint placements. Use synchronized elapsed time on warmed isolated requests with three repeated measurements. The measured region includes GPU input preparation, model calls and the choice. Tokenization, model loading, corpus search and external serving are outside the measurement. The mirrored route batches two endpoint presentations in one forward launch. Compare the quality-latency frontier without equating methods merely because they have similar token counts. [MuSiQue release](https://github.com/StonyBrookNLP/musique) · [Frozen panel](../../results/general_benchmarks_overnight/musique-manifest.json)
- **TabMWP source routing.** Take 128 grade-five through grade-eight questions, reserving the previous TabMWP question IDs, donor IDs and title/header groups. Keep a same-topic archive as the distractor. Run a typed route that accepts exactly one record matching a caller-provided table ID, with the same marker-free solver after routing. Verify order invariance, rejection of missing and duplicate IDs, and token-for-token equivalence to the current-only prompt before generation. Compare answer accuracy with direct reading of current and archived tables in both orders. This measures the benefit available when the caller supplies a correct ID. Learning to infer that ID from text is a separate question. [TabMWP release](https://github.com/lupantech/PromptPG) · [Prior exact-length control](TABMWP_ARCHIVE_NOISE.md) · [Frozen panel](../../results/general_benchmarks_overnight/tabmwp-manifest.json)

Report all matched seeds and untouched runs. Keep success rates paired by source family and timing summaries paired by request. Use the results to choose the next control, including a sharper latency comparison when routes have overlapping latency distributions.

## Fresh TabMWP results

| Supplied context | Untouched Llama | KAM | Dot |
| --- | ---: | ---: | ---: |
| Typed selection of the current table | 47.66% | 50.78% | 49.48% |
| Archive before current | 30.47% | 29.95% | 21.88% |
| Archive after current | 28.12% | 27.86% | 24.48% |

The archive-last drop from the typed route was 22.92 points for KAM and 25.00 for dot. The change in the KAM–dot gap was +2.08 points with a 95% family-bootstrap interval of −4.72 to +8.85. All seven checkpoints reproduced the current-only input tokens on all 128 queries, and the router rejected the missing-ID and duplicate-ID controls. The exact training-source audit found zero normalized overlaps across both fresh fixture sets. [Paired summary](../../results/general_benchmarks_overnight/tabmwp-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/tabmwp-outcomes.json.gz) · [Scheduler receipt](../../results/general_benchmarks_overnight/tabmwp-scheduler.tsv) · [Source audit](../../results/general_benchmarks_overnight/source-overlap.json)

Adding the archive last lost 106 previously correct KAM outcomes and 113 dot outcomes across their three seeds, while gaining 18 and 17. The grader extracted a candidate answer for all 106 KAM losses and 112 of the 113 dot losses. That rules out missing extraction as the main explanation for the score drop under this grader. [Numeric outcomes](../../results/general_benchmarks_overnight/tabmwp-outcomes.json.gz)

The length follow-up reserves the earlier control donor IDs and substitutes a different-topic archive of identical Llama token length on 83 of these 128 questions. Use the original two batch groups and retain the unchanged rows to check output parity. The comparison measures the effect of changing the archive contents at fixed input shape. [Frozen length-control panel](../../results/general_benchmarks_overnight/length-manifest.json)

### Fresh exact-length replication

| Archive context on the 83 matched questions | Untouched | KAM | Dot |
| --- | ---: | ---: | ---: |
| Current only | 60.24% | 66.27% | 64.26% |
| Same-topic archive last | 37.35% | 38.55% | 34.14% |
| Different-topic archive last | 55.42% | 53.82% | 55.02% |

Across 63 source groups, changing the archive improved accuracy by 15.26 points for KAM [4.39, 25.93] and 20.88 for dot [9.96, 31.82]. The paired change in the KAM–dot gap was −5.62 points [−13.18, +2.22]. The current table, question, archive position and token length were fixed. Each checkpoint reproduced the 45 unchanged outputs exactly. Archive-content interference replicated in both arms. [Paired summary](../../results/general_benchmarks_overnight/length-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/length-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/length-scheduler.tsv)

The exact normalized-text check found none of the 83 new donor title-and-table strings in the frozen adapter training sources. This audit covers the adapter data and the specified exact-match procedure. [Donor overlap audit](../../results/general_benchmarks_overnight/length-source-overlap.json)

## Live MuSiQue results

After the tokenizer visibility check, 128 questions in 48 relation families remained in the publisher's training stratum. The held-out development stratum retained six questions and is reported separately as a small check. The earlier study's question IDs, normalized wordings, relation families and supporting titles were reserved in both strata.

| Route on the 128-question stratum | KAM / dot, all four correct | KAM / dot p50 | KAM / dot p95 | KAM / dot, all four correct within 60 ms |
| --- | --- | --- | --- | --- |
| Direct | 15.10% / 19.01% | 34.20 / 33.87 ms | 55.00 / 54.70 ms | 15.10% / 19.01% |
| Online sequential | 65.36% / 72.92% | 54.38 / 53.66 ms | 74.67 / 74.09 ms | 44.01% / 49.74% |
| Mirrored, two views batched | 76.56% / 74.48% | 56.25 / 55.99 ms | 85.23 / 84.97 ms | 61.20% / 61.72% |

The latency columns average each checkpoint's warmed single-request percentiles over the three fixed fit seeds. Accuracy under a budget requires four correct presentations, each within the deadline. At 40 ms the faster direct route delivered more correct questions than either heavier route. At 60 ms, mirrors led this deadline comparison. At 150 ms, all successful presentations of all three methods fit. Scope the conclusion to these fixed candidate pools and in-process GPU execution.

As a follow-up, a single offset fitted only on the prior panel adjusted the A/B margin using the median midpoint across reversed labels. On this fresh stratum, calibrated direct accuracy rose to 19.27% KAM and 20.31% dot. The offset uses the same first forward pass. The paired gains on both arms have intervals spanning zero, so the data still favor testing cheaper ways to select the earlier passage. [Live summary](../../results/general_benchmarks_overnight/musique-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/musique-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/musique-scheduler.tsv)

### One-pass lexical routing follow-up

Rerun direct with the same frozen panel, then compare a CPU BM25 choice between the two equally truncated earlier passages followed by one GPU endpoint call. BM25 uses the question and the actual display-order texts, breaking ties by the supplied order. Confirm first-hop accuracy across both earlier positions, assess all-four endpoint success, and include the CPU rank operation in the same warmed elapsed-time scope. This is a fixed-pool test of a cheap first-stage selector. [Frozen comparison](../../results/general_benchmarks_overnight/lexical-manifest.json)

| Route on the 128-question stratum | All four correct, KAM / dot | p50, KAM / dot | p95, KAM / dot | All four correct within 40 ms, KAM / dot | Within 60 ms, KAM / dot |
| --- | --- | --- | --- | --- | --- |
| Direct | 15.10% / 19.01% | 34.16 / 33.87 ms | 54.90 / 54.82 ms | 10.94% / 12.50% | 15.10% / 19.01% |
| BM25 then one endpoint call | 68.23% / 75.78% | 31.23 / 30.48 ms | 41.32 / 41.09 ms | 54.17% / 61.20% | 68.23% / 75.78% |

BM25 found the verified earlier passage in both orders on 121 of 128 questions. The endpoint calls processed about 405 prompt tokens instead of 490. Untouched Llama rose from 7.81% to 52.34% on all-four accuracy. For the lexical route, KAM minus dot was −7.55 points [−11.44, −3.95] across question families conditional on the three measured training seeds. These were paired in-process timings on fixed candidate pools. The earlier sequential and mirrored routes were timed in a separate batch of Slurm jobs. [Paired summary](../../results/general_benchmarks_overnight/lexical-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/lexical-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/lexical-scheduler.tsv)

### Mirroring with a shorter context

Time three routes on the same GPU for each matched checkpoint. Compare BM25 followed by one endpoint view, mirroring with both earlier passages, and BM25 followed by two mirrored endpoint views in one launch. The first two routes control for the extra endpoint view and for dropping an earlier passage. Keep the four-presentation accuracy and 40/60 ms deadlines. Check first-hop agreement across the two lexical routes. This is a mechanism and latency comparison on the same frozen panel. [Frozen comparison](../../results/general_benchmarks_overnight/short-mirror-manifest.json)

### First-hop recall with native distractors

Keep the same visibility-checked questions and paired earlier candidates. Add up to six or 18 distinct nonsupporting paragraphs from each question’s original MuSiQue candidate pool, selected by a frozen hash of the question and paragraph identities. Exclude both endpoints and the paired earlier candidates from the additions. Compare BM25 recall at one, two and four in the forward and reverse order, using published full paragraphs throughout this CPU study. Report the count recovered in both orders and describe the effective pool sizes. This isolates first-hop retrieval before spending GPU time on the larger pools. [Audit script](../../scripts/audit_kam_musique_native_lexical.py)

| Candidate pool | Verified first hop at rank 1 | Within rank 2 | Within rank 4 |
| --- | ---: | ---: | ---: |
| Two paired candidates | 121 / 128 | 128 / 128 | — |
| Eight with native distractors | 91 / 128 | 112 / 128 | 122 / 128 |
| Up to 20 with native distractors | 77 / 128 | 94 / 128 | 114 / 128 |

The widest pool held 20 candidates in 121 cases, 19 in six and 18 in one. Every count requires recovery in both orders. Top two leaves room for a reranker to recover another 17 first hops over top one on this panel, provided it keeps the first hops BM25 already gets right. [Rank outcomes](../../results/general_benchmarks_overnight/native-lexical.json)

The GPU follow-up uses the widest pools. Compare BM25 top one against the same BM25 top two reranked by each frozen Llama checkpoint. Both routes then show the selected premise with the same two endpoints. Measure all-four endpoint accuracy, first-hop accuracy, and joint verified-first-hop and endpoint success at 40, 60, 100 and 150 ms. Compute BM25, prompt tokenization for model calls, launch and selection inside the timed region, and clip displayed documents to 512 tokens ahead of the timer. This compares two serving paths from fixed native pools. It uses the same questions as the CPU study and treats the published development stratum separately. [Locked GPU fixture](../../results/general_benchmarks_overnight/native-rerank-manifest.json)

## Physical batch retirement probe

Use the 32 hash-first questions from the fresh TabMWP split and the verified current-table prompt. Compare batch widths eight and 32. The fixed run keeps all cache rows until the last request stops. The retirement run checks every 16 steps and selects the still-active cache rows once at least one quarter have stopped. Keep both the backbone and residual dot caches aligned. Time each mode twice in ABBA order on the same GPU after a fixed-width warmup. Count scheduled token slots, active decode steps and complete token/EOS parity. Treat a speed change as usable at a measured width only if every token trace agrees. [Probe](../../scripts/probe_kam_batch_retirement.py)

| Batch width | Arm | Elapsed time reduction | Token slots removed | Exact traces |
| ---: | --- | ---: | ---: | ---: |
| 8 | KAM | 7.83% | 41.99% | 85 / 96 |
| 8 | Dot | 18.95% | 42.13% | 88 / 96 |
| 32 | KAM | 33.82% | 63.02% | 83 / 96 |
| 32 | Dot | 36.70% | 60.84% | 81 / 96 |

The probe failed the all-token parity requirement at both widths. The untouched backbone also had differences, matching 25 of 32 traces at width 32 and 28 of 32 at width eight. Treat these as timing and fidelity results for an experimental path. Keep fixed batches as the reference. [Retirement summary](../../results/general_benchmarks_overnight/retirement-summary.json) · [Numeric traces](../../results/general_benchmarks_overnight/retirement-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/retirement-scheduler.tsv)

Replay the same batches once per mode and verify the previously measured token-change and EOS flags. Grade the changed completions with the shared mathematical equivalence protocol. Count gains and losses separately by width and seed. This determines whether observed text changes affect verified answers in the probe; the exact-output gate remains in force. [Grade audit](../../scripts/grade_kam_retirement_audit.py)

The replay reproduced the registered mismatch flags, output lengths and EOS outcomes across all seven checkpoints. At width eight, every mathematical grade agreed despite 23 changed traces across the checkpoints. At width 32, KAM lost one verified answer across 96 outcomes. Dot gained one and lost one across 96. Untouched Llama gained two across 32. The width-32 timing gain came with both token differences and task-level changes. Keep fixed-batch generation as the reference. [Paired grade audit](../../results/general_benchmarks_overnight/retirement-audit-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/retirement-audit-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/retirement-audit-scheduler.tsv)

### Vocabulary projection for fixed-choice routing

Compute the same final hidden states and compare the full vocabulary projection plus an A/B gather against a linear projection with those two weight rows. Freeze a hash-selected sample of 24 MuSiQue pairs that passed the token-visibility checks, including the three published development pairs. Compare every placement for singleton and two-view mirrored requests. Time warmed full and sliced passes in counterbalanced ABBA order. Preallocate input and label tensors and slice the weights before timing. Report p50 and p95 timing, both selected logits and agreement on the final decision including ties. Promote the optimization for this particular route only if all decisions match and the latency reduction is consistent across matched devices and checkpoints. [Probe](../../scripts/probe_kam_restricted_readout.py) · [Frozen panel](../../results/general_benchmarks_overnight/readout-manifest.json)

## Public release clocks

Join stable releases from six public projects on the exact version string. Use the GitHub release publication timestamp and the earliest currently archived non-yanked PyPI file upload. Restrict both timestamps to 2019–2025. These are two recorded publication channels. The study uses a pinned snapshot of versions present in both sources, including historical times at which the two views differed.

Give the model the same three records and cutoff twice, changing which publication column the question names. Include a fourth option for an unavailable answer. Rotate all answer slots. Compare ordinary model choice, the same logits restricted to records eligible under the specified clock, and the deterministic latest-eligible baseline. Report both-clock and all-placement accuracy separately on paired cases where the gold record changes, where it agrees and where no displayed record is eligible. The availability cases test the boundary of the displayed archive. Treat project-macro rates as descriptive across six projects. [GitHub release API](https://docs.github.com/en/rest/releases/releases#list-releases) · [PyPI JSON API](https://github.com/pypi/warehouse/blob/main/docs/user/api/json.md) · [Frozen release fixture](../../results/general_benchmarks_overnight/clock-manifest.json)

On 61 cases where the target changed between the two publication channels, both fitted arms had zero cases correct across both channels and every placement. Presentation-level accuracy was 42.69% KAM and 41.94% dot. Restricting the same logits to versions eligible under the requested clock raised all-placement accuracy to 59.02% and 53.55%. Presentation-level accuracy rose to 91.73% and 90.92%. The remaining model errors after masking are among eligible choices, so the mask and the latest-version decision should be evaluated separately. The deterministic latest-eligible ordering recovers the gold on all displayed snapshots. [Release-clock summary](../../results/general_benchmarks_overnight/clocks-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/clocks-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/clocks-scheduler.tsv)
