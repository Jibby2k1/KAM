# BF16 radial features at fixed FlashAttention width

Using balanced metric coordinates and a two-lane key bias reduced the depth-512 gradient error on every run across three fresh seeds and two GPU types. The combined encoding preserved long-sequence throughput. Its depth-512 answer accuracy improved over original BF16 Flash on all three L4 seeds and on one of the three B200 seeds.

## Feature encodings

The existing augmented implementation packs the full diagonal metric and inverse bandwidth squared into the query and stores the negative key energy in one feature. In the balanced formulation both query and key coordinates are multiplied by \(\sqrt{M}/\sigma\), with feature preparation performed in FP32 before conversion to BF16. The optional two-lane bias places the rounded energy and the rounded residual into separate key features whose paired query features are both one. All Flash variants used the same padded width of 24. A local FP64 math-backend fixture reproduced the eager forward pass and all parameter gradients at absolute and relative tolerance \(10^{-12}\).

On B200 and L4, six modes began from the same 13.52-million-token checkpoint. Each of three seeds supplied a matched stream of 128 updates at depths 20 and 128 and disjoint 512-question tests at each depth. All input fingerprints and the checkpoint hash matched across devices. Gradient probes use eager FP32 as the reference. Timing covers the full update with clipping and fused AdamW, with 24 CUDA event measurements after five warm updates. [Runner](../../scripts/benchmark_kam_radial_precision.py) · [Audit](../../scripts/analyze_kam_radial_precision.py) · [Raw records](../../results/nonlinear_retrieval_natural/radial_precision_v1/)

## Depth 512

| GPU | Eager FP32 | Eager BF16 | Original Flash | Split bias | Balanced | Balanced and split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B200 | 90.89% | 90.30% | 91.41% | 89.58% | 90.17% | 91.15% |
| L4 | 90.89% | 88.93% | 89.58% | 89.71% | 89.19% | 91.08% |

Values are mean joint address-and-payload accuracy across three continuation seeds. The combined encoding changed the number correct out of 512 relative to original Flash by −4, −5, and +5 on B200 and +2, +8, and +13 on L4. Median relative L2 gradient error at depth 512 went from 0.1845 to 0.0802 on B200 and from 0.1797 to 0.0798 on L4. At depth 128 the median worsened from 0.2754 to 0.4081 on both devices, with essentially unchanged B200 accuracy versus original Flash and a 0.46-point improvement on L4.

| GPU | Eager BF16 step p50 | Original Flash p50 | Balanced and split p50 | Speedup versus eager BF16 |
| --- | ---: | ---: | ---: | ---: |
| B200 | 18.76 ms | 6.01 ms | 6.09 ms | 3.08× |
| L4 | 231.34 ms | 12.72 ms | 12.11 ms | 19.11× |

Timing is for batch 32 and 1,025 tokens. The combined Flash path used about 320 MB of extra peak allocated GPU memory, compared with 4,805 MB for eager BF16. The short continuations share one pretrained checkpoint and use generated data. Longer seeded training and revision-sequence transfer are the remaining quality checks. HPG jobs 52961065 and 52961066 generated these records.
