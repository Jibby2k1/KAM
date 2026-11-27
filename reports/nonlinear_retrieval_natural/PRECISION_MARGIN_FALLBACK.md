# Decision parity with calibrated mixed-precision fallback

On L4, a BF16-first fallback preserved both FP32 argmax decisions across 19,680 measured observations. It reran 50.9–68.1% of real archive lookups in FP32 and took longer than all-FP32 in every timed synthetic depth. The tested policy traded throughput for confidence and did not give a fast exact path on these inputs.

The model used the same three mixed radial checkpoints and mean-normalized signed-log publication coordinate as the archived clock work. For each seed and depth, 192 fresh synthetic episodes set separate thresholds for key and payload. The threshold was twice the maximum absolute logit error against the FP32 singleton terminal row for that head. When either BF16 top-two gap was inside its threshold, the policy recomputed only the flagged episodes with eager FP32. This bound is calibrated empirically from distinct fixtures. Its zero observed disagreements support parity on the tested panels. New inputs can exceed the calibration error.

| Panel | Observations | BF16 decisions matching FP32 | FP32 reruns | Fallback decisions matching FP32 |
| --- | ---: | ---: | ---: | ---: |
| Dallas, depth 16 | 3,504 | 3,410 | 2,367 | 3,504 |
| Dallas, depth 48 | 3,504 | 3,402 | 1,784 | 3,504 |
| Eurostat, depth 16 | 3,456 | 3,371 | 2,352 | 3,456 |
| Eurostat, depth 48 | 3,456 | 3,315 | 2,147 | 3,456 |
| Procedural, corrected depth 1,024 | 576 | 552 | 364 | 576 |
| Procedural, original depth 1,024 | 576 | 525 | 554 | 576 |
| All measured panels | 19,680 | 18,958 | 12,946 | 19,680 |

Counts are paired within each checkpoint and panel, with repeated requests under two real clock recodings. The equality is a decision-fidelity metric and preserves FP32 errors as well as successes. At depth 1,024 in the ordinary synthetic panel, full BF16 Flash took 1.48 ms, BF16 prefix with FP32 terminal row 1.32 ms, eager FP32 terminal row 20.64 ms, and calibrated fallback 22.33 ms. Recomputing the terminal row alone retained 529/576 FP32 decisions, four more than full BF16 Flash. This localizes most precision disagreements to the computations before that row under this checkpoint and workload.

HPG L4 array 53003781. [Calibration audit and paired outcomes](../../results/nonlinear_retrieval_natural/precision_margin_fallback_v1/) · [Terminal attention and hard selection](FINAL_ROW_ATTENTION.md)
