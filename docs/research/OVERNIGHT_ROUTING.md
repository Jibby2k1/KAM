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

The length follow-up reserves the earlier control donor IDs and substitutes a different-topic archive of identical Llama token length on 83 of these 128 questions. Use the original two batch groups and retain the unchanged rows to check output parity. The comparison measures the effect of changing the archive contents at fixed input shape. [Frozen length-control panel](../../results/general_benchmarks_overnight/length-manifest.json)

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

## Physical batch retirement probe

Use the 32 hash-first questions from the fresh TabMWP split and the verified current-table prompt. Compare batch widths eight and 32. The fixed run keeps all cache rows until the last request stops. The retirement run checks every 16 steps and selects the still-active cache rows once at least one quarter have stopped. Keep both the backbone and residual dot caches aligned. Time each mode twice in ABBA order on the same GPU after a fixed-width warmup. Count scheduled token slots, active decode steps and complete token/EOS parity. Treat a speed change as usable at a measured width only if every token trace agrees. [Probe](../../scripts/probe_kam_batch_retirement.py)

| Batch width | Arm | Elapsed time reduction | Token slots removed | Exact traces |
| ---: | --- | ---: | ---: | ---: |
| 8 | KAM | 7.83% | 41.99% | 85 / 96 |
| 8 | Dot | 18.95% | 42.13% | 88 / 96 |
| 32 | KAM | 33.82% | 63.02% | 83 / 96 |
| 32 | Dot | 36.70% | 60.84% | 81 / 96 |

The probe failed the all-token parity requirement at both widths. The untouched backbone also had differences, matching 25 of 32 traces at width 32 and 28 of 32 at width eight. Treat these as timing and fidelity results for an experimental path. Keep fixed batches as the reference. [Retirement summary](../../results/general_benchmarks_overnight/retirement-summary.json) · [Numeric traces](../../results/general_benchmarks_overnight/retirement-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/retirement-scheduler.tsv)

## Public release clocks

Join stable releases from six public projects on the exact version string. Use the GitHub release publication timestamp and the earliest currently archived non-yanked PyPI file upload. Restrict both timestamps to 2019–2025. These are two recorded publication channels. The study uses a pinned snapshot of versions present in both sources, including historical times at which the two views differed.

Give the model the same three records and cutoff twice, changing which publication column the question names. Include a fourth option for an unavailable answer. Rotate all answer slots. Compare ordinary model choice, the same logits restricted to records eligible under the specified clock, and the deterministic latest-eligible baseline. Report both-clock and all-placement accuracy separately on paired cases where the gold record changes, where it agrees and where no displayed record is eligible. The availability cases test the boundary of the displayed archive. Treat project-macro rates as descriptive across six projects. [GitHub release API](https://docs.github.com/en/rest/releases/releases#list-releases) · [PyPI JSON API](https://github.com/pypi/warehouse/blob/main/docs/user/api/json.md) · [Frozen release fixture](../../results/general_benchmarks_overnight/clock-manifest.json)

On 61 cases where the target changed between the two publication channels, both fitted arms had zero cases correct across both channels and every placement. Presentation-level accuracy was 42.69% KAM and 41.94% dot. Restricting the same logits to versions eligible under the requested clock raised all-placement accuracy to 59.02% and 53.55%. Presentation-level accuracy rose to 91.73% and 90.92%. The remaining model errors after masking are among eligible choices, so the mask and the latest-version decision should be evaluated separately. The deterministic latest-eligible ordering recovers the gold on all displayed snapshots. [Release-clock summary](../../results/general_benchmarks_overnight/clocks-summary.json) · [Numeric outcomes](../../results/general_benchmarks_overnight/clocks-outcomes.json.gz) · [GPU receipts](../../results/general_benchmarks_overnight/clocks-scheduler.tsv)
