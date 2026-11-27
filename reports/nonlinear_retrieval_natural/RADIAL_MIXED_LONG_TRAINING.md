# Radial training with 1,025-token steps

Including 1,025-token steps made fused radial attention 2.28× faster than eager BF16 on L4 and 1.07× faster on B200 in the full matched training loop. Extra peak allocated memory fell from 607–626 MB to 53 MB. All four training modes scored below 1.2% on the generated 128- and 512-record joint tests, so those tests primarily exposed a learning failure in this curriculum.

## Fixed-token mixed curriculum

Used the same three initialization seeds as the archive-only training study. Four modes shared 3,200 updates and 13,516,800 tokens. Each run cycled through archive-shaped procedural histories at depths 16 and 48, and ordinary mixed generated histories at depths 128 and 512. The 512-record batches had four examples. The corresponding input lengths were 33, 97, 257, and 1,025 tokens. Short-depth banks, initial weights, and archive test panels were identical to the short-curriculum study. The change also reallocates tokens across the two training sources. [Short-curriculum comparison](RADIAL_ARCHIVE_TRAINING.md)

| Device and training mode | Training path wall time | Earlier archive joint, 48 | Later archive joint, 48 | Generated joint, 512 |
| --- | ---: | ---: | ---: | ---: |
| B200 eager FP32 | 18.06 s | 85.36% | 80.21% | 0.78% |
| B200 eager BF16 | 19.55 s | 84.69% | 78.13% | 0.72% |
| B200 original BF16 Flash | 19.42 s | 85.63% | 81.25% | 0.52% |
| B200 balanced split BF16 Flash | 18.31 s | 86.93% | 81.51% | 0.52% |
| L4 eager FP32 | 34.99 s | 85.31% | 81.25% | 0.65% |
| L4 eager BF16 | 31.69 s | 85.26% | 80.73% | 0.78% |
| L4 original BF16 Flash | 13.68 s | 85.99% | 81.51% | 0.78% |
| L4 balanced split BF16 Flash | 13.89 s | 84.43% | 79.69% | 0.59% |

Quality is joint address-and-record correctness through a common eager FP32 forward path. At depth 48 every hardware and precision combination lost archived joint accuracy versus its own short-curriculum run. The balanced split losses were 0.99 and 1.56 points on the earlier and later B200 cohorts, and 4.38 and 3.65 on L4. The best generated 512-record panel reached four out of 512 correct for the balanced split path. The mixed curriculum changed both source and length exposure, so the cause of the generated-task failure remains open.

The training clock includes bank selection, input encoding, forward and backward, clipping, and optimizer updates. It excludes bank generation and evaluation. Memory is the extra PyTorch allocation above resident state during the fit. All four arms used the same label and inference checks on each device. HPG arrays 52966868 and 52966869 generated the results. [Raw and summarized results](../../results/nonlinear_retrieval_natural/radial_mixed_long_training_v1/) · [Cross-curriculum audit](../../scripts/analyze_kam_radial_mixed_long_training.py)
