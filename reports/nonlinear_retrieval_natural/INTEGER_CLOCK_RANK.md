# Packed integer sorting for address-local clock rank

Packing the address, eligibility and two integer clocks into one stable sort key sped up rank preparation on L4 and RTX PRO 6000. At depth 1,024 on L4, rank alone fell from 0.357 to 0.276 ms. The fused radial forward including clock preprocessing moved from 1.805 to 1.782 ms. The whole-path benefit was small after the larger attention optimization.

Packing preserves lexicographic direction across four fields. Address is the high field, followed by eligibility, reverse effective time, and reverse publication time. Each clock uses a signed 22-bit range. Equal-time pairs share their earliest within-address rank. The prototype assumes integer ticks from a trusted source with sufficient range and precision. The full input uses the original numerical offsets for normalization. General clocks still use stable multi-key sorting. A separate timed arm reconstructs ticks from the synthetic float inputs, whose grid is known.

| Timed mixed-future depth 1,024, four requests | L4 original | L4 packed | RTX original | RTX packed |
| --- | ---: | ---: | ---: | ---: |
| Address rank alone | 0.357 ms | 0.276 ms | 0.305 ms | 0.226 ms |
| Guard and rank preprocessing | 0.559 ms | 0.499 ms | 0.461 ms | 0.394 ms |
| Complete fused dot read | 1.463 ms | 1.429 ms | 1.032 ms | 0.995 ms |
| Complete fused radial read | 1.805 ms | 1.782 ms | 1.400 ms | 1.363 ms |

Each device verified both alias and fact preprocessing on 22,560 archive comparisons, including candidate depths, query symbolizations and fits. All prepared tensors were identical. In a separate synthetic sweep, all 2,304 paired fused outputs per GPU also matched bitwise. The typed metadata requires a separate integer tensor, adding about 131 KB to a depth-1,024 batch of four in the prototype. The table reports medians of three synchronized seed measurements, each using 28 repeats with fixed GPU-ready inputs. The timing excludes the cost of building or transferring metadata.

HPG L4 and RTX arrays 53018018 and 53018019. [Measured timing and parity](../../results/nonlinear_retrieval_natural/integer_clock_rank_v1/) · [Fused FP32 reference](GUARDED_FP32_SDPA.md)
