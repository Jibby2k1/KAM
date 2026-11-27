# Per-shape CUDA Graphs for shuffled radial updates

Capturing an update graph for each of three sequence lengths preserved parameters, optimizer moments, final losses, and held-out outcomes exactly on L4 and B200. With the depth order shuffled and the learning rate changing every update, median update time fell from 4.12 to 2.78 seconds on L4 and from 5.47 to 1.22 seconds on B200.

Every trial starts from the same trained checkpoint and uses 768 identical update examples, with three input lengths at 41, 257, and 1,025 tokens. The same seed fixes the dataset, shuffled depth order, per-length sample sequence and cosine rate. The selective kernel uses eager FP32 at the shorter lengths and balanced Flash BF16 for the long length. The host dispatches the graph matching each upcoming depth. A separate device cursor for each depth indexes that depth's sample list, and a global device cursor indexes the rate. The eager reference reads the same samples and rates in the same order. Both arms were repeated within each allocation, with independent seeds and rotated arm order.

| GPU | Eager updates | Per-shape graph updates | Eager setup + updates | Graph setup + updates | Paired training speedups |
| --- | ---: | ---: | ---: | ---: | ---: |
| L4 | 4.12 s | 2.78 s | 4.17 s | 3.08 s | 1.43–1.61× |
| B200 | 5.47 s | 1.22 s | 5.78 s | 1.49 s | 4.35–4.67× |

The medians are taken across seed-level trial medians. Estimated L4 break-even from separate setup and update timers ranged from 51 to 196 updates. The B200 estimate was at most 90 updates in these allocations. First-use setup varies by arm order and warmed CUDA state. Median extra peak allocated bytes were approximately 282–283 MB in all four arms, although reserved memory differed. The graph captures the same sequence of operations within each depth and preserves changes in rate and ordering between depths. The test covers three fixed batch shapes, so unseen shapes would require an eager path or a new capture.

HPG arrays 52986494 and 52986495 tested three seeds per device. [Optimizer and outcome parity](../../results/nonlinear_retrieval_natural/radial_shape_graphs_v1/) · [Randomized shape benchmark](../../scripts/benchmark_kam_radial_shape_graphs.py) · [Fixed-cycle graph replay](RADIAL_TRAINING_GRAPH.md)
