# Hard chronological visibility and within-address rank

Hard admission and within-address rank complemented each other on real between-release cutoffs. With identical clock normalization, mixed dot strict unemployment lookup improved from 45.75% with rank alone to 80.03% with the hard guard and rank. Industry improved from 60.33% to 91.20%. A guard alone gave lower deep correction accuracy than the corresponding unguarded model.

The guard places eligible records before the query and puts records after either requested clock at the causal suffix. It preserves each group’s incoming order. The query reads only the prefix, and positional features are disabled. Every arm scales clocks from available records only. The rank feature attaches to eligible facts, using order by effective time and then publication time within the workspace and entity. Ranking after the guard does not change the supplied rank values. Each model receives 3,200 updates and 10,556,800 input tokens from the same mixed training schedule. The hard policy keeps the full padded shape during training.

| Mixed decoder and features | Dallas, depth 48 | Unemployment, depth 48 | Industry, depth 48 | Corrections, depth 1,024 |
| --- | ---: | ---: | ---: | ---: |
| Dot, timestamps | 51.34% | 2.69% | 0.12% | 25.78% |
| Dot, timestamps and rank | 60.33% | 45.75% | 60.33% | 97.33% |
| Dot, hard clock | 32.91% | 33.39% | 45.25% | 18.49% |
| Dot, hard clock and rank | 60.64% | 80.03% | 91.20% | 95.96% |
| Radial, timestamps | 56.16% | 2.52% | 0.61% | 38.35% |
| Radial, timestamps and rank | 58.53% | 65.28% | 81.83% | 94.92% |
| Radial, hard clock | 22.09% | 23.41% | 29.95% | 0.59% |
| Radial, hard clock and rank | 58.30% | 82.47% | 91.17% | 93.49% |

Adding hard admission when rank is present improved dot's unemployment result by +34.29 points [29.77, 39.67] and industry's by +30.87 [25.00, 37.36]. For radial the effects were +17.19 [7.81, 23.70] and +9.35 [5.12, 13.54]. Dallas paired intervals included zero for both changes. On the smaller 2025 panel, the same dot comparison reached 90.28% from 53.47% in unemployment and 78.70% from 40.51% in industry. Those figures span only two release quarters. Rank already provides strong performance with another mean-clock normalization, so the matched differences above isolate guard effects under available-only scaling.

Changing unavailable addresses and payloads left all 384 tested hard-guard logits identical across seeds, geometries and guarded policies. Dot update time was 14.15 seconds for timestamps, 15.68 for rank alone, and 16.48 for both guard and rank, including per-update preprocessing. The guard assumes reliable typed eligibility fields. Ambiguous dates require an explicit uncertainty policy before hard admission. [Coarse-date ambiguities](REVISION_CLOCK_RESOLUTION.md) · [Timestamp-preserving rank](ADDRESS_LOCAL_CLOCK.md)

HPG L4 array 53011213. [Paired archive and depth outcomes](../../results/nonlinear_retrieval_natural/typed_clock_guard_v2/)
