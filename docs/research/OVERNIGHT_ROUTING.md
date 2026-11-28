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

## Physical batch retirement probe

Use the 32 hash-first questions from the fresh TabMWP split and the verified current-table prompt. Compare batch widths eight and 32. The fixed run keeps all cache rows until the last request stops. The retirement run checks every 16 steps and selects the still-active cache rows once at least one quarter have stopped. Keep both the backbone and residual dot caches aligned. Time each mode twice in ABBA order on the same GPU after a fixed-width warmup. Count scheduled token slots, active decode steps and complete token/EOS parity. Treat a speed change as usable at a measured width only if every token trace agrees. [Probe](../../scripts/probe_kam_batch_retirement.py)

## Public release clocks

Join stable releases from six public projects on the exact version string. Use the GitHub release publication timestamp and the earliest currently archived non-yanked PyPI file upload. Restrict both timestamps to 2019–2025. These are two recorded publication channels. The study uses a pinned snapshot of versions present in both sources, including historical times at which the two views differed.

Give the model the same three records and cutoff twice, changing which publication column the question names. Include a fourth option for an unavailable answer. Rotate all answer slots. Compare ordinary model choice, the same logits restricted to records eligible under the specified clock, and the deterministic latest-eligible baseline. Report both-clock and all-placement accuracy separately on paired cases where the gold record changes, where it agrees and where no displayed record is eligible. The availability cases test the boundary of the displayed archive. Treat project-macro rates as descriptive across six projects. [GitHub release API](https://docs.github.com/en/rest/releases/releases#list-releases) · [PyPI JSON API](https://github.com/pypi/warehouse/blob/main/docs/user/api/json.md) · [Frozen release fixture](../../results/general_benchmarks_overnight/clock-manifest.json)
