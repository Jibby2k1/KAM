# Flash fallback by query position

The layer-selected fallback recovered more measured attention on distant evidence while achieving the same retrieval accuracy as the periodic layer mask. This follow-up instead lets a small set of query positions use the complete causal prefix in every layer. [Distant-evidence experiment](DISTANT_EVIDENCE_FALLBACK.md)

## Fixed comparison

- Compare the final 32 queries, the final 256 queries, and a 256-query window immediately before the final 256. Keep the protected eight-block shortlist on other positions in all 32 layers. The preceding window controls for the number of dense queries and has a shorter available prefix.
- Skip the window in the Triton sparse pass. Fill just those outputs with fused SDPA and a lower-right causal bias against the prefix ending at that window. Match the slice to full causal Flash on the GPU and require unchanged sparse output outside it. [PyTorch causal bias](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.bias.causal_lower_right.html)
- Reuse the same three frozen checkpoints and the same 4,096-token holdout, with the previous all-Flash and all-layer protected quality controls. Report PG-19 perplexity and mirrored MuSiQue retrieval. For runtime, interleave both controls and the three windows on one HPG RTX PRO 6000. Measure batch-one prefill and incremental peak allocation at 512, 2,048, 4,096 and 8,192 tokens.

The final-window policies apply to both datasets. PG-19 scores the final 1,024 targets, so part of its scored continuation directly uses the dense window. The independent transfer panel will preserve the same PG-19 targets at both context lengths.

## Result on the 4K holdout

| Full-prefix attention queries per layer | PG-19 perplexity ↓ | Mirrored MuSiQue ↑ | 4K p50 ↓ | 8K p50 ↓ |
| --- | ---: | ---: | ---: | ---: |
| All queries, Flash | 10.2098 | 68.44% | 194.99 ms | 447.59 ms |
| None, protected shortlist | 11.2023 | 53.33% | 186.53 ms | 406.44 ms |
| Final 32 | 11.1809 | 52.40% | 187.51 ms | 408.08 ms |
| Final 256 | 11.0012 | 61.67% | 189.05 ms | 411.56 ms |
| Preceding 256 | 10.9792 | 54.27% | 188.98 ms | 411.30 ms |

The final 256 improved mirrored retrieval by 7.40 points [+2.50, +12.17] over the preceding window and by 8.33 points [+3.55, +13.33] over the fully protected shortlist. The gain over the preceding window appeared in all three seeds. Their NLL difference was +0.00200 nats/token [−0.00094, +0.00473]. Compared with Flash, the final window lost 6.77 retrieval points [−10.50, −2.95] and raised mean NLL by 0.07466 [0.06843, 0.08167]. Intervals resample relation families and books with seeds fixed.

The final 256 used 3.05% less 4K prefill time and 8.05% less 8K prefill time than Flash. Peak incremental PyTorch allocation at 4K and 8K was 469.14 and 935.17 MiB for all policies. The synthetic right-aligned Flash check matched the complete Flash readout within 0.00196 and preserved the sparse outputs outside each window bit for bit. The 8K benchmark here timed inference only.

The result supports prioritizing the answer-facing region when budgeting dense attention. The final 256 is the selected transfer arm because it improved mirrored retrieval over the shorter tail while remaining faster than Flash at 4K. Lock it before evaluating the independent length comparison. [Independent transfer protocol](QUERY_WINDOW_TRANSFER.md)

## Reproduction

Run [the query-window evaluator](../../scripts/kam_query_window_flash.py) as `ROOT SOURCE WORK smoke`, `evaluate INDEX` for indices 0–8, and `benchmark`. Use the prior registered Llama weights, checkpoint directory and frozen inputs. The [summarizer](../../scripts/summarize_kam_query_window.py) replays the outcomes with the backbone and first-block archives. [Numeric summary](../../results/general_benchmarks_query_window/summary.json) · [Replay archive](../../results/general_benchmarks_query_window/outcomes.json.gz) · [HPG jobs](../../results/general_benchmarks_query_window/scheduler.tsv)
