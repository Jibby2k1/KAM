# Independent transfer at 4K and 8K

Confirm the query-window policy selected on the preceding 4K holdout without retuning its budget. Compare it with forced Flash and the all-layer protected shortlist. [Query-window selection experiment](QUERY_WINDOW_FLASH.md)

## Fixed panel

- **PG-19.** Start from all 28 previously unused official test books. Retain every book with at least 8,193 Llama tokens. The 4K condition uses the final 4,097 tokens of the same 8,193-token window and both conditions score the same 1,024 targets. Books are the paired unit. [PG-19](https://github.com/google-deepmind/pg19)
- **MuSiQue.** Freeze 128 additional two-hop questions. Exclude question IDs reserved in earlier studies, and IDs, relation families and supporting titles from the preceding 320-question and development candidate pools. Use both A/B orders and identical endpoints across widths. Extend context with more of the same ordered distractor stream. Preserve the earlier passage and the literal endpoint-visibility checks. [MuSiQue](https://github.com/StonyBrookNLP/musique)
- **Controls.** Evaluate the same three frozen Llama checkpoints at both widths. Interleave Flash, the all-layer protected shortlist and the selected final-query window on the same RTX PRO 6000, including all routing and model work. Hash-lock the inputs and policy before evaluation. Audit normalized support and foil passages for exact adapter-training overlap.
- **Decision.** Report paired differences versus Flash and the protected shortlist at both lengths. Treat evidence about end-to-end quality and latency separately. Use the relative effect at 8K versus 4K to assess length sensitivity on the matched prompts.
