# Dot and radial retrieval under the same L4 training-time budget

On held-out procedural histories with 1,024 records per bank, optimized dot led at 25 seconds of training. The models were close at 45 seconds. At 60 seconds, depth-selective radial reached 56.64% joint accuracy and depth-selective dot reached 40.63%. Radial led by 14.45–19.14 percentage points across all three paired seeds, even though optimized dot processed more updates.

All modes start from matched shared parameters and consume prefixes of the same presealed data stream. The first 9,600 updates follow short-first training over 16, 48, 128 and 512 records per bank, followed by the same per-depth mix in shuffled order. Eager dot uses FP32 throughout. The selective dot and radial modes use eager FP32 for short steps and BF16 FlashAttention at 1,025 tokens, using the respective score formulations. All evaluation uses common eager FP32 inference, with distinct held-out simulator histories. After warmup, the time includes data assembly, forward and backward passes, and optimizer updates. Evaluation and setup are excluded. Budgets are checked every 120 updates, so all arms can exceed their nominal cutoff slightly.

| Update budget | Dot eager, joint 1,024 | Dot selective, joint 1,024 | Radial selective, joint 1,024 |
| ---: | ---: | ---: | ---: |
| 25 s | 31.38% | 34.90% | 19.01% |
| 45 s | 34.64% | 43.49% | 42.71% |
| 60 s | 38.15% | 40.63% | 56.64% |

At the last checkpoint, median measured training times were 60.38, 60.19, and 60.48 seconds respectively. The optimized dot arm completed a median of 17,040 updates, versus 11,280 for radial. The paired radial-minus-optimized-dot 95% intervals on 256 independent held-out questions per seed were [+12.50, +25.78], [+7.42, +21.09] and [+7.42, +21.48] points. At 512 records, radial scored 71.29%, versus 54.69% for optimized dot.

At 4,800 updates, optimized dot used a median 16.99 seconds versus 27.96 for eager dot, with 33.07% versus 31.25% joint accuracy at depth 1,024. The same-initialization attention probe preserved all 16 address-and-value decisions per seed, with FP32-reference gradient relative L2 error of 0.55–0.62%. These are distinct numerical and training-throughput checks.

In every seed the later 60-second optimized-dot checkpoint had fewer exact long-history answers than its 45-second checkpoint under the constant post-warmup learning rate. The training protocol holds data order fixed by update count, so a faster arm sees more repeated draws from its finite training banks. The comparison measures useful progress under this stream and optimizer, including this plateau. A matched learning-rate control is needed before attributing the late behavior solely to the score geometry.

The original pilot ended early for one optimized-dot seed after it consumed all presealed updates before 60 seconds. The reported HPG array 52983743 reran all three seed comparisons with the same initial schedule and more independently shuffled continuation blocks. [Timing and per-example outcomes](../../results/nonlinear_retrieval_natural/curriculum_wallclock_v2/) · [Runner](../../scripts/run_kam_curriculum_wallclock.py) · [Matched FP32 trajectories](RADIAL_DOT_CURRICULUM.md)
