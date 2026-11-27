# Deterministic CUDA Graph training for radial attention

On NVIDIA L4, capturing the depth-selective radial training loop reduced the median duration of 768 updates from 4.00 to 2.78 seconds. In deterministic mode, every parameter, both AdamW moment vectors, final losses, and held-out decisions matched across eager execution and capture for three seeds. A device-side replay cursor also preserved these results under a cosine learning-rate schedule. Its additional speed benefit on L4 was negligible.

## Matched training

Each run began from the same capable [radial checkpoint](RADIAL_DISPATCH.md). A cycle uses sequence lengths 41, 257, and 1,025 with their fixed batch shapes. Each path ran the identical 256 cycles and candidate indices for a given seed. Capture includes tensor packing, the forward pass, backward pass, gradient clipping, and AdamW. Actual-shape warmup and graph capture were measured separately. Before timing, the parameters and allocated optimizer state were reset in place. The optimizer step count was audited for every active parameter.

The constant-rate control used a fixed scalar. The cosine experiment used a device tensor for the learning rate and compared three replay paths. The host-staged graph copied each cycle's selections and rates into fixed storage. The cursor graph indexed the resident training schedule and advanced the cursor inside the graph. Each mode was repeated within the same allocation, with execution order rotated across seeds.

| L4 workload | Kernel policy | Eager | Host-staged graph | Device-cursor graph |
| --- | --- | ---: | ---: | ---: |
| Constant learning rate | Eager FP32 at all lengths | 10.69 s | 10.39 s | — |
| Constant learning rate | FP32 short and balanced Flash BF16 long | 4.00 s | 2.78 s | — |
| Cosine learning rate | Eager FP32 at all lengths | 10.64 s | 10.38 s | 10.37 s |
| Cosine learning rate | FP32 short and balanced Flash BF16 long | 4.09 s | 2.78 s | 2.78 s |

Values are medians across the three seed-level medians, each based on two runs. The selective path's constant-rate speedup ranged from 1.40× to 1.51× by seed. The parameter and outcome comparisons are within a kernel policy. The [kernel dispatch study](RADIAL_DISPATCH.md) measures differences between precision policies.

## Deterministic B200 replication

The three-seed B200 comparison matched parameters, both AdamW moment vectors, final losses, and held-out decisions exactly within both kernel policies. From the same capable checkpoint and at the same 768 updates, capture reduced median all-FP32 time from 5.34 to 1.63 seconds and the depth-selective loop from 5.78 to 1.20 seconds. Paired speedups ranged from 3.23× to 3.42× for all-FP32 and 4.47× to 5.15× for the selective path. The selective capture setup took 0.16–0.26 seconds across seeds. The B200 result uses the constant learning rate and host staging. The graph-side cursor was tested under a cosine rate on L4.

## Numerical diagnostic

Before deterministic algorithms were enabled, the selective path produced relative weight differences of 0.00735 between eager and graph, 0.00996 between two eager runs, and 0.00734 between two graph runs at the same L4 seed. The pure FP32 path was bitwise stable across all four runs. This identifies run-to-run variability within the selective path under that configuration. The three-seed deterministic runs enabled PyTorch deterministic algorithms and a cuBLAS workspace configuration of `:4096:8`. [PyTorch CUDA Graph and reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/cuda.html)

One nondeterministic B200 pilot reduced the pure FP32 loop from 4.44 to 1.73 seconds with exact parameter, optimizer, and outcome parity. Its selective path was faster after capture and exhibited numeric drift. The B200 pilot used one seed and preceded the deterministic cross-device comparison.

Arrays 52973623 and 52974922 ran the deterministic L4 comparisons. Jobs 52972475 and 52972476 ran the same-path diagnostic and the short deterministic pilot. Job 52971604 ran the first B200 comparison and array 52973624 ran the three-seed deterministic B200 replication. [Seed-level replay and audit](../../results/nonlinear_retrieval_natural/radial_training_graph_v1/) · [Benchmark runner](../../scripts/benchmark_kam_radial_training_graph.py)
