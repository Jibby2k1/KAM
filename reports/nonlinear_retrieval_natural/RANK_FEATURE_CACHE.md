# Cache deterministic rank features for repeated training histories

Caching within-address clock ranks saved time on L4, B200 and RTX PRO 6000 and preserved weights, AdamW moments, sampled losses and every held-out decision. Combining the cache with per-shape CUDA Graphs cut B200 mixed dot training from 19.41 to 3.03 seconds including setup in the matched graph experiment.

Each training scene and permutation is fixed before drawing the seeded minibatches. The cache stores one unsigned byte per serialized token, with 255 reserved for tokens whose feature stays unchanged. Decoding applies the same reciprocal as runtime processing. The cache uses 1,064,960 bytes across both training depths and banks. Verification compares the entire transformed input tensor on matched draws from each depth. Training uses the same 3,200 updates, the same 10,556,800 tokens, and all 1,536 held-out episode outcomes per queried panel family across three seeds.

| GPU and geometry | Eager update-only, online → cached | Independent graph experiment, eager online → graph online → graph cached including setup |
| --- | ---: | ---: |
| L4 dot | 16.53 → 14.66 s | 14.96 → 5.81 → 5.53 s |
| L4 radial | 22.61 → 20.50 s | 20.97 → 8.42 → 8.10 s |
| B200 dot | 20.55 → 18.50 s | 19.41 → 3.54 → 3.03 s |
| B200 radial | 26.68 → 24.57 s | 26.41 → 5.39 → 4.86 s |
| RTX PRO 6000 dot | 12.29 → 10.65 s | 11.30 → 3.68 → 3.20 s |
| RTX PRO 6000 radial | 16.55 → 15.04 s | 15.69 → 5.18 → 4.77 s |

The earlier eager setup builds the rank cache in 0.08–0.40 seconds. Every paired seed saves training time after accounting for that setup. The graph experiment packs the same rows into a single gather, fixes the same FP32 kernels and rate sequence, and crosses eager and captured training with online and cached features. It verifies equivalent input batches, all weights and moments, final losses and held-out outcomes. The initial online capture failed at deterministic CUDA scatter. Gathering the inverse permutation gives the same rank values and allows capture. The absolute timings should be compared within each experiment. The cache targets repeated immutable histories. New histories or changed cutoffs require fresh rank calculation.

HPG L4, RTX and B200 arrays 53008820/21/23 and 53009786/87/88. [Eager audit](../../results/nonlinear_retrieval_natural/rank_feature_cache_v1/) · [Graph crossover and staged source hashes](../../results/nonlinear_retrieval_natural/rank_cache_graph_v2/) · [Effect of the cached feature](ADDRESS_LOCAL_CLOCK.md)
