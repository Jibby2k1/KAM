# Training order can restore guarded radial snapshot selection

The guarded radial decoder recovered 11 of 1,536 exact depth-1,024 correction choices with interleaved training. Moving the first 800 shallow updates ahead of deep examples raised recovery to 1,532 at the same input token count, update count, and wall time. Moving every shallow example first also restored deep accuracy and gave markedly worse Dallas archive performance.

Each run uses the same 3,200 updates, 10,556,800 input tokens, per-depth sample streams and seeded parameters as the fused FP32 training comparison. Every arm uses the same per-shape CUDA Graph kernel on L4 and the same learning-rate schedule by update number. The interleaved arm cycles through nominal depths 16, 48, 128 and 512. The first-half shallow arm alternates 16 and 48 for 800 updates, using the first 400 batches of each shallow depth. It then shuffles the outstanding depth slots, preserving batch order within each depth. The all-shallow-first arm alternates 16 and 48 for 1,600 updates and follows with the original relative order of depths 128 and 512. The control reproduced the archived loss and all held-out flags exactly on every seed.

| Geometry and exact training order | Inclusive fitted time, median | Deep corrections / 1,536 | Dallas / 3,504 | Unemployment / 3,456 | Industry / 3,456 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dot, interleaved | 8.72 s | 1,536 | 1,869 | 2,852 | 3,182 |
| Dot, first half of shallow batches | 8.77 s | 1,536 | 1,992 | 2,764 | 3,150 |
| Dot, all shallow batches | 8.82 s | 1,536 | 1,672 | 2,562 | 3,079 |
| Radial, interleaved | 11.17 s | 11 | 841 | 255 | 268 |
| Radial, first half of shallow batches | 11.19 s | 1,532 | 1,789 | 2,566 | 3,035 |
| Radial, all shallow batches | 11.17 s | 1,535 | 172 | 1,853 | 2,323 |

All three radial interleaved fits retrieved the deep alias with perfect precision on these answerable test cases. Their failure was snapshot selection. The first-half schedule gained 1,521 procedural joint matches and lost zero relative to the interleaved radial control. The randomized remainder may contribute to the result. Both shallow-first paths retain the same demonstrations overall, including archive-shaped examples, which only occur at depths 16 and 48 in this curriculum. Their large differences on Dallas establish that training exposure totals alone do not explain the transfer. Archive counts include three fits and two serializations of each query, with depth 48 held fixed. Timings exclude validation and fixture setup.

HPG L4 array 53017133. [Paired outcomes and order hashes](../../results/nonlinear_retrieval_natural/typed_guarded_sdpa_curriculum_v1/) · [Matched kernel and capture study](GUARDED_SDPA_TRAINING.md)
