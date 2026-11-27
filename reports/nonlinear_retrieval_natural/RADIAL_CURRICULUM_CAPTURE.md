# From-scratch curriculum training with per-shape CUDA Graphs

Per-shape CUDA Graphs reproduced complete from-scratch training runs on HPG L4 and B200. Within each GPU, geometry and seed, the captured path matched eager parameters, optimizer moments, final loss and held-out decisions byte for byte. Across three seeds, B200 captured update time was 6.0× faster for dot and 5.3× for radial using the ratio of median timings.

Each arm starts from the same untrained state for its seed. Short-first curriculum order, per-depth samples, batch size and all 19,200 cosine-scheduled rates are shared. Per-shape draws and the rate are indexed by separate device cursors. Eager and captured modes use the same selective kernel within a geometry. Geometry is compared at the same update budget. Capture and eager order rotate across allocations.

| GPU | Geometry | Eager updates | Captured updates | Eager with setup | Captured with setup | Depth 1,024 accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L4 | Dot | 71.41 s | 42.05 s | 71.49 s | 42.50 s | 47.27% |
| L4 | Radial | 102.84 s | 66.57 s | 102.93 s | 67.11 s | 83.85% |
| B200 | Dot | 108.64 s | 18.04 s | 109.21 s | 18.39 s | 47.14% |
| B200 | Radial | 159.38 s | 30.04 s | 159.52 s | 30.83 s | 83.59% |

Times are medians and accuracies are means across seeds. Captured peak allocated memory was 245–248 MB above the measurement baseline, with 245–252 MB across both modes. Different HPG host and allocator states can affect eager dispatch timings, so compare capture with eager within each allocation. Device-to-device parity is not expected with different kernels and numerical hardware. The experiment fixes update budget and does not measure when each geometry reaches a given accuracy.

HPG arrays 52987721 and 52987722. [Measurements and parity checks](../../results/nonlinear_retrieval_natural/curriculum_capture_v1/) · [Training runner](../../scripts/run_kam_curriculum_capture.py) · [Summary script](../../scripts/analyze_kam_curriculum_capture.py) · [Earlier shuffled-depth capture](RADIAL_SHAPE_GRAPHS.md)
