# Independent transfer at 4K and 8K

Confirm the query-window policy selected on the preceding 4K holdout without retuning its budget. Compare it with forced Flash and the all-layer protected shortlist. [Query-window selection experiment](QUERY_WINDOW_FLASH.md)

## Fixed panel

- **PG-19.** Start from all 28 previously unused official test books. Retain every book with at least 8,193 Llama tokens. The 4K condition uses the final 4,097 tokens of the same 8,193-token window and both conditions score the same 1,024 targets. Books are the paired unit. [PG-19](https://github.com/google-deepmind/pg19)
- **MuSiQue.** Freeze 128 additional two-hop questions. Exclude question IDs reserved in earlier studies, and IDs, relation families and supporting titles from the preceding 320-question and development candidate pools. Use both A/B orders and identical endpoints across widths. Extend context with more of the same ordered distractor stream. Preserve the earlier passage and the literal endpoint-visibility checks. [MuSiQue](https://github.com/StonyBrookNLP/musique)
- **Controls.** Evaluate the same three frozen Llama checkpoints at both widths. Interleave Flash, the all-layer protected shortlist and the selected final-query window on the same RTX PRO 6000, including all routing and model work. Hash-lock the inputs and policy before evaluation. Audit normalized support and foil passages for exact adapter-training overlap.
- **Decision.** Report paired differences versus Flash and the protected shortlist at both lengths. Treat evidence about end-to-end quality and latency separately. Use the relative effect at 8K versus 4K to assess length sensitivity on the matched prompts.

## Independent result

Twenty-three of the remaining official PG-19 test books had enough tokens for an 8K window. The 128 MuSiQue questions all came from the publisher's training split and were held out from this experiment's tuning. The exact passage check found no normalized support or foil matches in the adapter training sources. The selected policy was Flash for the final 256 queries at each layer.

| Context | Policy | PG-19 perplexity ↓ | Mirrored MuSiQue ↑ | Prefill p50 / p95, ms ↓ | Incremental peak, MiB ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,096 | Flash | 11.0198 | 74.48% | 194.30 / 194.65 | 469.14 |
| 4,096 | Protected shortlist | 12.2512 | 59.64% | 185.65 / 186.07 | 469.14 |
| 4,096 | Final 256 | 11.9322 | 66.41% | 188.28 / 188.69 | 469.14 |
| 8,192 | Flash | 10.8083 | 55.47% | 447.46 / 447.88 | 935.17 |
| 8,192 | Protected shortlist | 12.3469 | 50.78% | 406.17 / 406.63 | 935.17 |
| 8,192 | Final 256 | 11.9701 | 44.53% | 411.31 / 411.87 | 935.17 |

The final window improved NLL relative to the protected shortlist at both lengths. The gain was 0.02639 nats/token at 4K and 0.03099 at 8K. It also reduced prefill p50 by 3.10% and 8.08% relative to Flash.

Mirrored retrieval relative to the protected shortlist shifted from +6.77 points [−2.36, +15.89] at 4K to −6.25 [−15.22, +2.82] at 8K. The paired change was −13.02 points [−24.22, −1.59]. Compared with Flash, the window lost 8.07 points [−14.36, −2.08] at 4K and 10.94 [−21.61, −0.26] at 8K. The interaction relative to Flash had a wide interval [−15.25, +9.56]. All retrieval intervals resample relation families with the three seeds held fixed.

The fixed final window gives a measured prefill savings and a length-sensitive retrieval tradeoff. The 4K retrieval improvement over the protected shortlist was directionally consistent with the preceding study. It did not transfer to 8K on these matched prompts. Flash remains the quality reference at both lengths.

## Reproduction

Use [the preparer](../../scripts/prepare_kam_attention_transfer.py) to acquire and checksum the public PG-19 books, reserve prior MuSiQue sources, encode the two widths, and run the normalized passage audit. [The evaluator](../../scripts/eval_kam_attention_transfer.py) uses the registered Llama weights and prior checkpoint directory, with `evaluate INDEX` for indices 0–8 and `benchmark` for interleaved timing. [The summarizer](../../scripts/summarize_kam_attention_transfer.py) replays the paired results and verifies the locked policy.

[Input manifest](../../results/general_benchmarks_attention_transfer/transfer-manifest.json) · [Selected policy](../../results/general_benchmarks_attention_transfer/routing-selection.json) · [Summary](../../results/general_benchmarks_attention_transfer/summary.json) · [Replay archive](../../results/general_benchmarks_attention_transfer/outcomes.json.gz) · [HPG jobs](../../results/general_benchmarks_attention_transfer/scheduler.tsv)
