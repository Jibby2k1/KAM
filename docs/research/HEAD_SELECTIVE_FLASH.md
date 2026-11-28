# Selecting dense attention by head group

At the same dense query-head budget, assigning dense attention to SmolLM2 key–value head groups improved held-book reconstruction over choosing whole layers. With four mixed layers, local projected error fell 23.00% and live perplexity fell from 13.99 to 13.91. Permitting more mixed layers took live perplexity to 13.82. Full dense reached 13.46. The Apple GPU prototype retained a prefill speedup versus full dense.

## Choose the mask on separate books

Use the same 16 model inputs and pinned SmolLM2-135M checkpoint as the omitted-summary study. Allocate eight books to calibration and eight to evaluation by hashed book ID. For frozen output measurements, compare two queries per sampled tile, at offsets zero and 31, on all 30 layers. Compute each query head's sparse error separately and project it through that head's slice of the attention output matrix. If $e_h$ is that projected error, the local squared error after choosing dense heads is $\|\sum_{h\in S}e_h\|^2$ where $S$ collects the remaining sparse heads. Its exact evaluation uses the dot products of all pairs in $S$. Sum the layer contributions and divide by each book's dense output energy.

Use dynamic programming to choose 135 dense layer–head pairs out of 270, with each sparse head keeping the same protected route. Compare alternating layers, calibrated whole-layer choices, sets of complete three-query-head groups sharing one key–value head, and unrestricted individual heads. Constrain the number of layers which mix dense and sparse heads in some arms. Fit the masks from calibration states, then hold them fixed for both frozen and live evaluation on the eight held-out books. The diagonal arm fits without cross-head error terms and is evaluated on the true summed error.

| Dense allocation | Mixed layers | Held-book normalized error | Relative improvement over calibrated whole layers [paired 95% interval] |
| --- | ---: | ---: | ---: |
| Alternating whole layers | 0 | 0.77973% | −451.77% [−540.90, −369.88] |
| Calibrated whole layers | 0 | 0.14131% | Reference |
| Complete groups, up to four mixed layers | 4 | 0.10882% | 23.00% [20.67, 25.85] |
| Complete groups, up to eight mixed layers | 8 | 0.08950% | 36.67% [35.35, 37.75] |
| Complete groups, unrestricted | 12 | 0.07824% | 44.64% [42.50, 46.18] |
| Individual heads, unrestricted | 22 | 0.04308% | 69.51% [66.83, 71.56] |
| Individual heads, diagonal fit | 23 | 0.04463% | 68.42% [65.93, 70.36] |

The four-mixed-layer arm met the prespecified local gate with improvement in all eight test books. The live comparison kept both group masks locked.

## Live perplexity and local prefill

Use the same 512 next-token targets per held-out book for each policy. During live prefill, each layer's input states come from the preceding layers under the same policy. The mixed layers run dense and sparse head subsets separately and combine them before the original output projection.

| Attention policy | Live perplexity | NLL change vs. calibrated whole layers [paired 95% interval] | 4K prefill p50 |
| --- | ---: | ---: | ---: |
| Native dense SDPA | 13.4597 | −0.03849 [−0.04948, −0.02777] | 3,655.45 ms |
| Protected shortlisting | 14.8437 | +0.05938 [+0.03558, +0.08685] | 2,437.45 ms |
| Calibrated whole layers | 13.9879 | Reference | 2,935.66 ms |
| Groups, up to four mixed layers | 13.9146 | −0.00526 [−0.00953, −0.00115] | 2,962.22 ms |
| Groups, unrestricted | 13.8176 | −0.01225 [−0.01785, −0.00704] | 2,563.79 ms |

The four-mixed-layer arm improved language loss at broadly similar prefill to the whole-layer schedule. The unrestricted groups recovered 31.82% of the calibrated whole-layer excess NLL above dense in this comparison. The timing uses one fixed 4,096-token prompt, Apple M5 PyTorch 2.10.0 MPS, FP32, two warm-ups per mode and nine synchronized interleaved trials, including shortlist construction, mixed-head dispatch, full backbone, and one-position vocabulary projection. Per-policy timings varied across trials. This is a result for the measured local PyTorch implementation. An RTX fused Flash baseline and task quality require a separate comparison.

[Masks and frozen holdout](../../results/general_benchmarks_pretrained_sparse_local/head-selection.json) · [Live losses and individual latency samples](../../results/general_benchmarks_pretrained_sparse_local/head-live.json) · [Calibration code](../../scripts/audit_kam_head_fallback.py) · [Live evaluation](../../scripts/eval_kam_head_fallback.py)
