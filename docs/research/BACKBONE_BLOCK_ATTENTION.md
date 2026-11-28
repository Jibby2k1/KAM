# Block-shortlisted attention inside the Llama backbone

Replace the attention operator in 8, 16 or 32 of Llama's 32 layers. Compare with forced Flash SDPA at depth zero. Use the same three frozen top-four-linear adapter checkpoints and the same 4,096-token PG-19 and MuSiQue inputs. The four residual adapters keep their exact top-four kernel at every depth. This measures an inference-time replacement of pretrained attention. [Prior adapter-only shortlist study](BLOCK_SHORTLIST.md)

## Fixed protocol

- **Operator.** Keep the existing projections, grouped-query layout, rotary positions and output projection. Represent each 64-token key block with four means, each computed over a contiguous 16-key group. The first query in each 32-token tile chooses a shortlist for that tile. Include the current and previous key blocks and score up to six older blocks by their best mean score. Use causal softmax over the original token keys in all selected blocks. Limit the operator to unpadded, uncached prefill. [Triton implementation](../../scripts/kam_backbone_block_attention.py)
- **Depth sweep.** At depth 8, replace layers 4, 8, …, 32 in one-based numbering. At depth 16, replace layers 2, 4, …, 32. At depth 32, replace every layer. Compare all depths against Flash with the same model weights and inputs. Use one shortlist budget of eight blocks so the experiment isolates replacement depth.
- **Fidelity.** Compare the new grouped-query kernel to Flash when it covers the entire causal prefix. Rebuild a sparse mask for an independently normalized Torch reference once the context exceeds the block budget. Perturb future queries, keys and values and verify that earlier routes and outputs are identical. Check the full backbone at 512 tokens with and without all 32 operators replaced, where the eight-block budget still covers all eligible keys.
- **Quality.** Score 147,456 targets from 48 PG-19 books and mirrored endpoint choice on 320 MuSiQue questions. Force Flash on every remaining dense backbone layer, including all layers of the control. Compare paired book NLL and mirrored retrieval for every depth against the new all-Flash runs. Hold the three adapter seeds fixed in bootstrap intervals. Report sensitivity to the one detected adapter-training passage overlap.
- **Performance.** Measure 512, 2,048, 4,096 and 8,192 tokens on an HPG RTX PRO 6000 using the same frozen model at each depth. Include key means, shortlist lookup, retained-key softmax and the complete model in each timed pass. Interleave 15 warmed batch-one trials per depth. Report synchronized wall p50 and p95, CUDA p50, and incremental peak PyTorch allocation over the resident model and inputs. The 8,192-token result uses concatenated book windows for timing.

Read the 4,096-token timing and quality together. A latency win accompanied by a large perplexity or retrieval loss does not establish a useful serving tradeoff.

## Full-model prefill on HPG

Times are synchronized wall p50 / p95 in milliseconds on an NVIDIA RTX PRO 6000 Blackwell Server Edition. Each timing includes shortlist construction and lookup in every selected layer, all other layers, and the same A/B readout. The last column gives the incremental peak allocated by PyTorch, which was identical at each depth. GPU events and individual trials are in the result archive.

| Tokens | Flash, 0 layers | 8 layers | 16 layers | 32 layers | Peak across all depths, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 34.293 / 34.777 | 34.242 / 34.758 | 34.300 / 34.589 | 34.351 / 34.869 | 61.36 |
| 2,048 | 97.663 / 97.705 | 97.095 / 97.293 | 96.972 / 96.993 | 96.098 / 96.339 | 236.12 |
| 4,096 | 196.591 / 197.278 | 194.293 / 194.863 | 192.141 / 192.698 | 187.904 / 188.547 | 469.14 |
| 8,192 | 449.583 / 451.155 | 439.071 / 440.743 | 429.123 / 430.466 | 408.241 / 409.707 | 935.17 |

At 4,096 tokens, the eight-, sixteen- and thirty-two-layer replacements reduced median prefill by 1.17%, 2.26% and 4.42%. At 8,192 tokens, the corresponding reductions were 2.34%, 4.55% and 9.20%. These timings cover one checkpoint, batch size and GPU type.

The first profiling run hit the GPU shared-memory limit in the 8K proxy dot product. The retry accumulates that dot product in 32-channel tiles and uses the same block-selection rule. The job ledger retains both attempts.

## Quality on the frozen 4K holdout

| Shortlisted backbone layers | Eligible token pairs retained | PG-19 perplexity ↓ | Paired NLL change, nats/token [95% CI] | Mirrored MuSiQue ↑ | Mirrored change, points [95% CI] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0, Flash | 100.00% | 10.2098 | — | 68.44% | — |
| 8 | 80.52% | 33.7581 | +1.196 [+1.159, +1.232] | 48.23% | −20.21 [−27.10, −13.59] |
| 16 | 61.04% | 174.3468 | +2.838 [+2.745, +2.929] | 52.50% | −15.94 [−22.40, −9.72] |
| 32 | 22.09% | 337.7396 | +3.499 [+3.439, +3.560] | 50.31% | −18.13 [−24.83, −11.75] |

The token-pair column counts original causal query–key pairs across all backbone layers and excludes proxy comparisons. Restoring the all-Flash control reproduced the prior aggregate NLL and mirrored accuracy for each checkpoint. Perplexity rose with replacement depth in all three seeds. The mirrored scores fell at every depth. Removing the one question with a detected adapter-training passage overlap changed the mirrored gaps to −20.27, −15.67 and −17.97 points. The intervals resample books and retrieval relation families with the three seeds fixed.

At 512 tokens, when the shortlist still covers every eligible key, replacing all 32 layers changed next-token NLL by −0.00213 and −0.00358 nats/token on the two integration windows. The Torch sparse-mask reference and future-token perturbation also passed. At 8K, a separate Torch implementation matched all 96 tested block routes and the corresponding readout differed by at most 0.00195.

## Research decision

The frozen backbone replacement achieved a measurable full-reader prefill speedup and left peak allocation flat. Even the lightest tested replacement roughly tripled perplexity and lost 20 points of mirrored retrieval. The earlier adapter-only shortlist preserved quality. That experiment used its own per-query routing and normalized over the four highest-scoring keys. Keep Flash as the serving reference. An efficiency claim requires a quality-preserving backbone design. Use per-layer attention-mass retention and a small anchor-block or layer-placement diagnostic to decide whether a learned routing study is warranted. [Adapter-only results](BLOCK_SHORTLIST.md)

## Reproduction

With the same registered Llama weights, adapter checkpoint directory and frozen tokenized inputs, run [the evaluator](../../scripts/eval_kam_backbone_shortlist.py) as `ROOT SOURCE WORK smoke`, `wide-check`, `benchmark`, and `evaluate INDEX` for indices 0–11. Run [the summarizer](../../scripts/summarize_kam_backbone_shortlist.py) with the archived outputs, prior input manifest, prior long-context outcomes, and prior source audit. The numerical archive preserves the original JSON output and can replay the published summary byte for byte. [Summary](../../results/general_benchmarks_backbone_shortlist/summary.json) · [Archive](../../results/general_benchmarks_backbone_shortlist/outcomes.json.gz) · [HPG jobs](../../results/general_benchmarks_backbone_shortlist/scheduler.tsv)
