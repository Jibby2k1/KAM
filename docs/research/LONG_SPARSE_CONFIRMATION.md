# Long-context confirmation of top-four live-token selection

Reuse the three step-256 top-four-linear and soft-linear Llama 3 checkpoints plus untouched Llama. The fit schedules, parameter counts and training exposure match within seed. The test data are new. This is an independent test of the trained models and a measured inference implementation comparison. [Matched training and prior holdout](LIVE_TOKEN_FACTORIAL.md)

## Evaluation protocol

- **Language modeling.** Use 48 official PG-19 test books with no overlap with the prior 24. Choose three disjoint 4,096-token input windows per book. Score the last 1,024 next-token targets in each window, for 147,456 targets. Report book-paired estimates and perplexity from weighted mean NLL. [PG-19](https://github.com/google-deepmind/pg19)
- **Retrieval.** Use 320 new MuSiQue two-hop questions, with 236 from the publisher's development split and 84 from its training split. Exclude 1,622 IDs reserved from prior Llama and title-field panels. Preserve the verified earlier passage, annotated endpoint and a native nonsupporting endpoint foil. Insert natural distractors between the earlier and endpoint passages to reach exactly 4,096 tokens. Keep evidence visible and exclude the accepted answer and anchor from the inserted background. A question is correct only if the model picks the verified endpoint in both A/B orders. Also report the 81 questions with a foil that mentions the anchor. The exact-match audit against adapter-training sources detected one support passage in the locked cohort. Report sensitivity to dropping that question. [MuSiQue](https://github.com/StonyBrookNLP/musique)
- **Inference.** Force PyTorch Flash SDPA for soft-linear attention. Use a tiled Triton exact top-four scorer and a gather over only the selected values for sparse. It still scans eligible keys, retaining four candidates per key tile. The two-stage implementation avoids storing the full score matrix and has tile-candidate scratch space. It has no sublinear lookup or KV-cache compression guarantee. Keep the Llama backbone and output projections unchanged. [Triton implementation](../../scripts/kam_exact_sparse_readout.py)
- **Quality parity.** Before scoring each trained checkpoint, compare its inference backend with the explicit selection implementation used during training. Use the first four books and first 24 locked questions. Allow at most one A/B choice difference in 48 presentations and 0.003 absolute NLL difference per window. If an arm misses the gate, use the explicit backend for its quality score and retain the inference discrepancies separately. After the first parity miss, run the full holdout with the explicit scorer for every matched checkpoint, reusing the runs that already fell back. This control measures the architecture difference with a shared backend.
- **Latency and memory.** Compare prefill and a single cached-history read for the adapter kernel at 512, 2,048, 4,096 and 8,192 keys. Repeat at the same context lengths for batch-one full-model prefill using the same frozen weights for both operators and an A/B readout. Interleave warmed measurements after compilation and tensor/model setup. Report CUDA time, synchronized wall time and the incremental PyTorch allocated peak above resident inputs and weights. The single-query test isolates the attention read from KV append and full-model decoding. [Kernel profiler](../../scripts/benchmark_kam_sparse_live_read.py) · [Full-reader evaluator](../../scripts/eval_kam_long_sparse.py)

Treat an out-of-sample gain with paired uncertainty separately from implementation speed. A quality–cost claim requires validated quality and a meaningful improvement in the full reader on a prespecified context length. If sparse selection retains quality but the exact scan costs more than fused dense attention, the result supports a scoped selection finding and leaves an efficiency claim open. [ICML 2027 contribution types](https://icml.cc/Conferences/2027/ContributionTypes)

## Held-out quality at 4,096 tokens

| Variant | PG-19 perplexity ↓ | MuSiQue endpoint correct in both orders ↑ | Mirrored-margin accuracy ↑ |
| --- | ---: | ---: | ---: |
| Untouched Llama | 11.1776 | 1.25% | 65.94% |
| Soft linear | 10.2550 | 4.90% | 68.02% |
| Top-four linear | 10.2098 | 8.54% | 68.44% |

The table uses the locked parity-gated scoring policy. All three top-four kernels passed the gate, with one A/B choice difference apiece on 48 audit presentations. Two soft-linear Flash runs passed. Seed 731211 used the original scorer after Flash changed three of its 48 audit choices. The standalone sparse kernel matched 39,485 of 39,488 BF16 output values exactly across five reference shapes. The largest absolute difference was 0.000122.

Top-four reduced PG-19 perplexity by 0.44%, or 0.00442 nats/token in mean NLL. The signed book-paired NLL difference was −0.00442 [−0.00521, −0.00366]. Per-seed differences were −0.00004, −0.00557 and −0.00766. The 95% intervals resample books or retrieval relation families with the three trained seeds held fixed.

| MuSiQue cohort | Questions | Both orders, soft linear | Both orders, top-four linear | Paired difference, points [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Locked panel | 320 | 4.90% | 8.54% | +3.65 [+2.23, +5.23] |
| Publisher dev | 236 | 4.24% | 7.34% | +3.11 [+1.60, +4.93] |
| Publisher train | 84 | 6.75% | 11.90% | +5.16 [+2.08, +8.73] |
| Anchor mentioned in both endpoints | 81 | 2.88% | 4.53% | +1.65 [0.00, +3.85] |
| Exclude detected adapter-training passage match | 319 | 4.60% | 8.25% | +3.66 [+2.25, +5.26] |

**Answer-order diagnostic.** Soft-linear chose A in 96.15% of presentations and top-four in 93.13%. To cancel a shared label offset, take the B-minus-A logit margin with gold in B and subtract the B-minus-A margin with gold in A. A positive difference selects gold. Under this post hoc two-pass readout, top-four led by 0.42 points [−1.22, +2.05]. The raw directional result survives removal of the one exact-match passage, but provides limited evidence of better evidence discrimination once the label position effect is controlled. Both metrics supply the verified earlier passage and a fixed two-endpoint pool.

**Common-scorer control.** The explicit training implementation reproduced the perplexity effect (10.2549 soft linear, 10.2098 top-four linear, 0.44% lower). Its paired NLL difference was −0.00441 [−0.00519, −0.00365]. Both-order retrieval was 5.10% soft linear and 8.44% top-four linear, a +3.33-point gap [+1.99, +4.82]. After subtracting swapped-order margins, the scores were 67.92% and 69.17%, a +1.25-point gap [−0.20, +2.76]. This control supports the small perplexity effect across backends. It leaves the retrieval advantage sensitive to the readout.

## HPG latency and memory

Both measurements used an RTX PRO 6000 Blackwell Server Edition with PyTorch 2.8.0, CUDA 12.8 and Triton 3.4.0. Each pair below is Flash / exact top-four. Peak memory measures live PyTorch allocations above the prefetched input and parameter footprint. This scope omits allocator reservations and memory owned by other libraries.

| Context | Full-reader wall p50, ms ↓ | Full-reader wall p95, ms ↓ | Full-reader peak, MiB ↓ |
| ---: | ---: | ---: | ---: |
| 512 | 34.079 / 34.105 | 34.252 / 34.290 | 61.36 / 61.36 |
| 2,048 | 97.459 / 97.474 | 98.913 / 98.718 | 236.12 / 236.12 |
| 4,096 | 193.318 / 193.575 | 193.549 / 193.872 | 469.14 / 469.14 |
| 8,192 | 443.363 / 444.054 | 443.900 / 444.529 | 935.17 / 935.17 |

| Context | Adapter prefill CUDA p50, µs ↓ | Adapter prefill peak, MiB ↓ | Single history read CUDA p50, µs ↓ | Single history read peak, KiB ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 24.29 / 32.21 | 0.59 / 0.31 | 23.87 / 32.11 | 3.5 / 1.5 |
| 2,048 | 29.60 / 55.50 | 8.53 / 4.25 | 23.62 / 31.50 | 6.5 / 2.5 |
| 4,096 | 47.82 / 119.60 | 8.81 / 16.50 | 24.19 / 31.20 | 10.5 / 4.5 |
| 8,192 | 173.50 / 317.50 | 1.13 / 65.00 | 27.18 / 31.30 | 18.5 / 8.5 |

Sparse adapter prefill took 2.50× the fused dense CUDA time at 4,096 tokens. Its temporary candidate buffers grew with the number of queries and key tiles. A single history read reduced temporary allocation at all four widths and still took longer. The Llama backbone dominated full-reader time and peak allocation, leaving no measured end-to-end savings. The full-reader numbers cover prefill and an A/B readout with caching disabled, at batch one for one checkpoint and GPU type.

## ICML claim

The result supports a narrow claim. After matched adapter tuning on the same frozen Llama backbone, exact top-four linear selection reduced PG-19 perplexity by 0.44% on 48 disjoint long-context books and the effect survived a common scorer. The effect varied by training seed and was nearly zero for one seed. The endpoint retrieval improvement depends on the readout, and the measured sparse value gather offers no end-to-end latency or memory savings over Flash. Hold the broader retrieval and efficiency claims. Prioritize testing whether a cheap candidate shortlist can avoid some key scores and deliver an end-to-end quality–cost improvement on independent data.

## Reproduction

Use the frozen [input manifest](../../results/general_benchmarks_long_sparse/manifest.json), [PG-19 object list](../../results/general_benchmarks_long_sparse/pg19-sample.json), and [reserved MuSiQue IDs](../../results/general_benchmarks_long_sparse/reserved-musique-ids.txt). Download the listed books from the PG-19 release and verify their hashes. The [preparation script](../../scripts/prepare_kam_long_sparse.py) uses `freeze` to select MuSiQue rows, then `inputs` to tokenize both datasets with the registered Llama checkpoint. The manifest fixes book blocks, evidence, foils, distractor order and the complete tokenized-input digest.

With the registered Llama bundle, prior adapter checkpoint directory and prepared input directory, run `eval_kam_long_sparse.py ROOT SOURCE WORK evaluate INDEX` for indices 0–6. Run `evaluate-reference` for the six trained checkpoints after their parity-gated evaluations. Run `profile` for the full-reader benchmark and `benchmark_kam_sparse_live_read.py WORK/kernel-results.json` for kernel fidelity and standalone timings. Use the [summarizer](../../scripts/summarize_kam_long_sparse.py) with the outcomes directory, frozen manifest, prior [checkpoint receipt archive](../../results/general_benchmarks_live_token/outcomes.json.gz), and [training source audit](../../results/general_benchmarks_long_sparse/source-audit.json). The results archive stores the original numeric JSON files under their file stems for byte-for-byte replay. [Numeric summary](../../results/general_benchmarks_long_sparse/summary.json) · [Results archive](../../results/general_benchmarks_long_sparse/outcomes.json.gz) · [HPG job ledger](../../results/general_benchmarks_long_sparse/scheduler.tsv)
