# Can fixed prefix protection recover distant retrieval?

Reserving the first four of eight shortlisted blocks covers the complete bridging mention on every independent MuSiQue presentation. It raises average PG-19 perplexity and gives less than one percentage point of mirrored retrieval improvement over an equal-sized set of background pins at both widths.

## Matched comparison

Hold three frozen Llama checkpoints, endpoints, native distractors and paired 4K/8K inputs fixed. Compare two prefix pins (0 and 1), four prefix pins (0 through 3), and four control pins (0, 8, 9, 10). All three preserve the current and preceding block and fill the remaining slots by the existing proxy score. Where pins overlap the current or previous block they cost no extra slot. The protected reference always preserves block zero. The final-window reference uses Flash on the last 256 queries in every layer.

The span audit verifies the same early passage tokens in both widths. The two-prefix policy guarantees 66.49% of earlier-passage tokens and all bridge tokens in 238 of 256 presentations. Four prefix pins guarantee 97.98% of passage tokens and the full bridge in 256 presentations. The control guarantees 9.72% of earlier-passage tokens and the full bridge in 28 presentations, all through the shared first block. Dynamic routes can recover more.

| Policy | 4K PG-19 PPL | 4K mirrored | 4K p50 | 8K PG-19 PPL | 8K mirrored | 8K p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash | 11.0198 | 74.48% | 193.17 ms | 10.8083 | 55.47% | 446.34 ms |
| Protected | 12.2512 | 59.64% | 184.50 ms | 12.3469 | 50.78% | 405.26 ms |
| Final-256 Flash | 11.9322 | 66.41% | 187.17 ms | 11.9701 | 44.53% | 409.76 ms |
| Two prefix pins | 12.3229 | 61.72% | 184.63 ms | 12.4732 | 48.18% | 404.59 ms |
| Four prefix pins | 12.8404 | 59.90% | 184.62 ms | 13.0144 | 47.14% | 404.79 ms |
| Four control pins | 12.7888 | 59.11% | 184.74 ms | 13.0593 | 46.61% | 405.12 ms |

Four prefix pins minus four control pins yielded +0.78 mirrored points at 4K [−5.78, +7.49] and +0.52 at 8K [−4.23, +5.43]. Two prefix pins minus the protected shortlist yielded +2.08 points at 4K [−3.71, +8.01] and −2.60 at 8K [−7.81, +2.56]. Relative to protected shortlisting, four prefix pins raised mean token NLL by 0.0470 at 4K [0.0315, 0.0630] and 0.0527 at 8K [0.0391, 0.0669]. Retrieval intervals resample question families. NLL intervals resample books. Both condition on the three seeds.

Quality reused 23 independent PG-19 books and 128 MuSiQue questions. The RTX PRO 6000 profile interleaved 15 batch-one trials per width and policy. Incremental allocated peak stayed at 469.14 MiB at 4K and 935.17 MiB at 8K. GPU parity passed for all three pin patterns against an independently assembled sparse causal mask with maximum BF16 output difference 0.0078125.

The experiment shows that guaranteed access to the early bridge alone is insufficient to close the measured Flash gap. Giving a narrower span to specific query positions can test whether static prefix pins discard useful competing blocks.

[Paired scores and timings](../../results/general_benchmarks_pinned_blocks/summary.json) · [Scheduler receipts](../../results/general_benchmarks_pinned_blocks/scheduler.tsv)
