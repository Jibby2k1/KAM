# Dot and radial attention under identical curricula and precision

On a synthetic bitemporal task, dot-product attention learned faster early in training. Radial attention overtook it by 9,600 updates. At 1,024 records per bank, with training capped at 512, final joint accuracy was 52.21% for short-first radial and 38.15% for short-first dot. Both used the same eager FP32 precision, per-depth records, optimizer, initial shared weights, and elapsed-clock inputs. The radial update loop took longer in this comparison.

## Matched protocol

We used the same three seed panels as the [short-first scheduling study](RADIAL_CURRICULUM_ORDER.md). The two model families had no positional encodings and used the same query-relative temporal features. Their initial common parameter hashes, sample indices, data-bank hashes, and evaluation panels matched by seed. The radial arm additionally learned a diagonal metric and bandwidth. Both training schedules used 39,307,200 tokens across the same 9,600 updates. Each schedule ran with eager FP32 for both geometries. The hardware and training policy are fixed within the comparison. The model family is the intervention.

| Joint accuracy | Dot interleaved | Dot short first | Radial interleaved | Radial short first |
| --- | ---: | ---: | ---: | ---: |
| 4,800 updates, depth 128 | 63.93% | 66.54% | 50.26% | 58.14% |
| 4,800 updates, depth 512 | 42.06% | 43.42% | 22.46% | 30.47% |
| 4,800 updates, depth 1,024 | 28.39% | 31.25% | 11.33% | 19.79% |
| 9,600 updates, depth 20 | 89.84% | 89.26% | 93.42% | 92.97% |
| 9,600 updates, depth 128 | 71.29% | 73.11% | 84.70% | 85.29% |
| 9,600 updates, depth 512 | 48.44% | 52.02% | 63.09% | 63.48% |
| 9,600 updates, depth 1,024 | 37.89% | 38.15% | 49.35% | 52.21% |

For short-first training at 1,024 records, final radial-minus-dot effects were +16.41, +12.50, and +13.28 percentage points. All corresponding per-seed paired question-bootstrap intervals excluded zero. The 4,800-update checkpoint favored dot at the three longer test depths under both schedules. Relative ordering therefore depends on training progress as well as model geometry.

The 9,600 update loops averaged 63.96–67.89 seconds for dot and 110.85–114.86 seconds for radial on L4. These are equal-exposure comparisons. The separate [selective precision study](RADIAL_DISPATCH.md) evaluates a cheaper radial implementation. A matched wall-clock comparison should use optimized kernels for both model families.

HPG array 52980531 ran six allocations, three seeds for each geometry. [Matched checkpoints and paired estimates](../../results/nonlinear_retrieval_natural/radial_dot_curriculum_v1/) · [Analysis](../../scripts/analyze_kam_curriculum_geometry.py)
