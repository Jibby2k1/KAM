# Does a block shortlist preserve exact top-four reads?

Compare a causal 64-token block shortlist with exact top-four selection and fused dense Flash SDPA. Reuse the three fitted top-four linear adapters and the locked 4,096-token holdout from the prior study. The holdout contains 48 disjoint PG-19 test books with 147,456 scored tokens and 320 position-controlled MuSiQue questions. Checkpoint and tokenized-input hashes bind the comparison to the original run. [Long-context protocol](LONG_SPARSE_CONFIRMATION.md)

## Locked comparison

- **Routing.** Represent each block by the means of its four contiguous 16-key groups. For each query, rank completed blocks by the largest dot product against their four means. Always include its current block. Apply the causal mask when scoring keys in that block. Rerank eligible keys in the selected blocks using their original dot products, then read the best four values. Budgets of four, eight and sixteen include the current block. Eight is the primary setting. [Triton implementation](../../scripts/kam_block_shortlist.py)
- **Exact-key audit.** Record routed key IDs and compare against an exhaustive stable FP32 dot-product ranking on real activations. Audit every 64th query starting at token index 31, for every adapter head and layer, on one window from each of the first four books and both label orders for the first four MuSiQue questions. Each budget is evaluated on activations produced by its own inference path. Report key recall, the fraction of head queries recovering the entire four-key set, and the effect of position. Verify the kernel on synthetic inputs that fit within the budget and check that perturbing future keys and values leaves earlier outputs and key IDs unchanged.
- **Quality.** Evaluate all three shortlist budgets on the same full holdout. Report PG-19 perplexity and the paired NLL difference against the prior exact sparse output. For MuSiQue, compare the B-minus-A margin with gold in B against the margin with gold in A. The difference must be positive. Report family-paired intervals conditional on the fixed seeds. Track both-order agreement and sensitivity to the one detected adapter-training passage overlap as secondary measures. [Quality evaluator](../../scripts/eval_kam_block_shortlist.py)
- **Cost.** Match GPU type, inputs, projected adapter head shape and, in the full reader, the same frozen top-four checkpoint. Interleave the three budgets, exact top-four and forced Flash SDPA after warmup. At 512, 2,048, 4,096 and 8,192 tokens, measure prefill and single-query history reads at the adapter boundary and batch-one full-reader prefill with A/B output heads. Time shortlist construction, block lookup, key scoring and value gathering together on every invocation. Report synchronized wall p50 and p95, CUDA p50 and incremental peak PyTorch memory above resident inputs and parameters. Full-reader prefill runs with caching disabled. The single-query history test rebuilds the shortlist summaries from resident keys.

Evaluate the eight-block setting first and use the four- and sixteen-block results to show the recall–cost tradeoff. Report whether any setting preserves exact-key recovery, held-out quality and a measured latency improvement over Flash. Changes in accuracy, memory and latency remain separate observations.

## Key recovery and held-out quality

| Readout | Eligible key scores at 4K | Exact-key recall ↑ | All four recovered ↑ | PG-19 perplexity ↓ | MuSiQue mirrored accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fused soft linear, prior run | 100% | — | — | 10.2550 | 68.02% |
| Exact top-four linear, prior run | 100% | 100% | 100% | 10.2098 | 68.44% |
| Shortlist, 4 blocks | 10.67% | 54.33% | 22.62% | 10.2103 | 68.23% |
| Shortlist, 8 blocks | 22.09% | 72.25% | 40.97% | 10.2083 | 68.33% |
| Shortlist, 16 blocks | 42.59% | 87.40% | 66.09% | 10.2077 | 68.85% |

Eligible key scores count original token keys and exclude proxy dot products. The audit spans 36,864 sampled head queries and 147,456 exact key IDs per budget across the three seeds. For eight blocks, all-four recovery was 84.75% in the first quarter of the context, 30.38% in the middle half and 18.37% in the last quarter. Even sixteen blocks recovered all four on only 37.91% of last-quarter queries.

Against exact top-four, eight blocks changed mean NLL by −0.00014 nats/token with a paired 95% interval of [−0.00034, +0.00006]. Mirrored accuracy changed by −0.10 percentage points [−0.90, +0.77]. Excluding the question with the detected adapter-training passage overlap gave a mirrored gap of −0.10 points [−0.94, +0.78]. The other budgets changed mirrored accuracy by −0.21 and +0.42 points respectively, with paired intervals spanning zero. Across budgets, the absolute change in perplexity from the exact readout was at most 0.021%. The large drop in exact-key recovery had little effect on these downstream metrics for the tested frozen adapters.

## HPG latency and peak memory

Profiling used an RTX PRO 6000 with PyTorch 2.8.0 and Triton 3.4.0. Adapter timings include constructing all proxy means and looking up a shortlist on every call. Full-reader timings include all model layers at batch one. Each row compares the five operators on the same inputs and weights. The quality row for fused soft linear above comes from the prior matched fit, whose quality used the explicit reference for one seed after its Flash parity check.

| Length | Flash, µs | Exact, µs | 4 blocks, µs | 8 blocks, µs | 16 blocks, µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 26.46 | 32.45 | 43.54 | 44.34 | 49.79 |
| 2,048 | 31.98 | 56.00 | 58.66 | 62.00 | 88.74 |
| 4,096 | 49.63 | 120.88 | 77.54 | 87.50 | 143.90 |
| 8,192 | 172.90 | 314.96 | 117.44 | 141.98 | 261.71 |

The table reports median CUDA time for adapter prefill. At 4,096 tokens, eight blocks were 27.6% faster than exact top-four and 76.3% slower than Flash. They used 1.03 MiB of incremental peak allocation, compared with 16.50 MiB for exact top-four and 8.81 MiB for Flash. At 8,192 tokens, four and eight blocks beat Flash by 32.1% and 17.9% in isolated CUDA time. Their allocation peaks were 1.56 and 2.06 MiB, compared with 1.13 MiB for Flash. The one-query history read with summary construction was slower than Flash at all lengths and budgets.

| Length | Flash p50, ms | Exact p50, ms | 4 blocks p50, ms | 8 blocks p50, ms | 16 blocks p50, ms | Full-reader peak, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 34.200 | 34.219 | 34.228 | 34.265 | 34.269 | 61.36 |
| 2,048 | 97.293 | 97.413 | 97.407 | 97.399 | 97.500 | 236.12 |
| 4,096 | 195.784 | 196.081 | 195.791 | 195.916 | 196.115 | 469.14 |
| 8,192 | 449.066 | 449.868 | 449.195 | 449.150 | 449.330 | 935.17 |

The table reports synchronized batch-one wall time and the incremental PyTorch allocation peak, which was identical across operators at each length. At 4,096 tokens, full-reader wall p95 was 196.23 ms for Flash and 196.46 ms for eight blocks. At 8,192, it was 450.44 and 450.52 ms. The profile measured one GPU type and one frozen checkpoint. Perplexity and retrieval use the locked 4,096-token holdout. The 8,192-token result measures runtime using concatenated book windows.

## Research decision

The mean-proxy shortlist is useful as an approximate-routing ablation. It preserved the measured downstream quality while substituting many exact key identities. It reduced isolated prefill cost relative to Flash at 8,192 tokens for the two smallest budgets. Full-reader latency and memory provided no serving advantage. This experiment supports a scoped separation between exact-key recovery and downstream quality. An ICML efficiency claim needs a measured improvement in the complete reader. Use the shortlist as a quality reference when testing designs that reduce work in a larger share of the model.

## Reproduction

Run [the evaluator](../../scripts/eval_kam_block_shortlist.py) with `ROOT SOURCE WORK smoke` for synthetic and causal checks, `evaluate INDEX` for indices 0–8, and `benchmark` for the single-GPU cost comparison. Use the prior study's same registered Llama bundle, adapter checkpoint directory and prepared inputs. Run [the summarizer](../../scripts/summarize_kam_block_shortlist.py) on the archived raw JSON files, prior input manifest, prior results archive and prior source audit. [Numeric summary](../../results/general_benchmarks_block_shortlist/summary.json) · [Replay archive](../../results/general_benchmarks_block_shortlist/outcomes.json.gz) · [HPG job ledger](../../results/general_benchmarks_block_shortlist/scheduler.tsv)
