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

## An evenly spaced remainder and independent held-out histories

With a deterministic evenly spaced remainder, 400 early shallow updates recovered 1,512/1,536 fresh corrections, and 800 recovered 1,535. The shuffled order without an explicit warmup recovered 391. These results isolate the gain from a shallow start without depending on a shuffled remainder. Moving another 400 shallow updates to the end after the same 400-step start reduced fresh radial performance to 16/1,536.

Every policy consumes the same four streams of 800 batches in order within each stream. The evenly spaced schedules reserve the shallow prefix and use the remaining fractions of each depth's batches as priorities to distribute the rest. The bookend variant reserves 400 shallow updates both first and last. The fresh synthetic set changes all generation seeds and includes no training episodes. Real archive panels remain the same to quantify the transfer cost.

| Radial order | Original deep corrections / 1,536 | Fresh deep corrections / 1,536 | Dallas / 3,504 | Unemployment / 3,456 | Industry / 3,456 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interleaved | 11 | 11 | 841 | 255 | 268 |
| Shuffle, no shallow start | 393 | 391 | 686 | 1,557 | 1,912 |
| 400 shallow then evenly spaced | 1,514 | 1,512 | 1,426 | 2,062 | 2,619 |
| 800 shallow then evenly spaced | 1,534 | 1,535 | 1,809 | 2,558 | 3,035 |
| 400 shallow, evenly spaced, 400 shallow | 18 | 16 | 758 | 413 | 565 |

The balanced 800 schedule used 11.35 seconds versus 11.25 for interleaving. The similarly warmed dot model recovered all fresh deep corrections and had higher archived counts than radial under the same schedule on these typed histories. All-shallow-first and bookend checks show that a late distribution of examples can change transfer even after shallow initialization. The settings compare order at fixed tokens, weights and data. They do not identify a universal shallow warmup length.

HPG L4 array 53018472. [Fresh procedural replay and archive outcomes](../../results/nonlinear_retrieval_natural/typed_guarded_sdpa_curriculum_v2/)
