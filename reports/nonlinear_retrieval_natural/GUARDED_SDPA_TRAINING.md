# Guarded FP32 training with eager and captured execution

On L4, fused FP32 plus CUDA Graphs reduced a fixed 3,200-update radial fit from 29.22 to 11.19 seconds including setup. On RTX PRO 6000, eager attention under capture took 7.68 seconds, compared with 8.55 for captured fused attention. Capture preserved weights and AdamW moments within each kernel. Changing the kernel caused finite-precision training drift, even though the fixed-checkpoint inference study preserved every hard choice.

The study starts from the same seeded parameters within each geometry. Every arm sees the same interleaved batch sequence at depths 16, 48, 128 and 512, totaling 10,556,800 tokens. Clock admission and address-local rank are applied within the training step. The guarded query reads only its eligible prefix. Deep synthetic inputs have up to two records per nominal depth plus one query token. Both kernels use PyTorch FP32 and deterministic operations. All fitted weights are evaluated with the same eager attention readout. Archive and procedural comparisons retain the same provenance within each seed.

| Device and geometry | Eager attention, eager loop | Eager attention, captured | Fused FP32, eager loop | Fused FP32, captured |
| --- | ---: | ---: | ---: | ---: |
| L4, dot | 17.41 s | 16.77 s | 11.75 s | 8.72 s |
| L4, radial | 29.22 s | 28.10 s | 18.12 s | 11.19 s |
| RTX PRO 6000, dot | 11.73 s | 5.50 s | 9.70 s | 6.94 s |
| RTX PRO 6000, radial | 16.03 s | 7.68 s | 14.86 s | 8.55 s |

The table uses medians across three independently initialized seeds per device. The captured path matched the eager loop's complete parameter and moment tensors, final loss, every archived correctness flag and every procedural correctness flag within all 12 device/geometry/kernel comparisons. Peak extra live allocation for radial was 508.0 MB with the fully eager path and 137.9 MB with captured fused FP32. Across kernels, relative weight differences ranged from 0.0148 to 0.0347 for dot and from 0.00010 to 0.00042 for radial. On L4, dot's strict unemployment result moved from 2,837 to 2,852 of 3,456 repeated archive cases and industry from 3,168 to 3,182, showing near agreement in aggregate with individual differences.

Changing training depth had a larger association than the kernel change. Under the long interleaved schedule, both L4 radial kernels chose the correct alias on all 1,536 depth-1,024 correction episodes and the snapshot on 11. Both dot kernels reached 1,536 joint choices. The RTX replication showed the same radial count and 1,535 for dot. The preceding shallow schedule at the same update and token budgets yielded 93.49% joint radial and 95.96% dot. The schedules redistribute training tokens and archive-shaped examples across depths, so the separate comparison does not isolate example order. The synchronized wall times exclude evaluation and raw fixture generation.

HPG L4 array 53015476 and RTX array 53015477. [Per-fit timings, paired outcomes and exact-state checks](../../results/nonlinear_retrieval_natural/typed_guarded_sdpa_train_v1/) · [Fixed-checkpoint inference](GUARDED_FP32_SDPA.md) · [Shallow hard-clock and rank training](TYPED_CLOCK_GUARD.md)
