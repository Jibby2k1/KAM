# Phase 6 Stage3 Router Scaling report

- Row outputs: **32**
- Passing rows: **32**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage3_router_scaling/hpg_runs_profile_scaling1`
- Manifest: `results/phase6/stage3_router_scaling/manifests/profile_hpg_38049475.jsonl`
- Artifact manifest: `results/phase6/stage3_router_scaling/hpg_runs_profile_scaling1/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: router_scaling=32
- `router`: approximate=8, chunked=8, exact=8, product_key=8
- `precision`: bf16=10, fp16=11, fp32=11

## Primary grouped summary

| router | precision | n | recall_at_k_against_exact | routing_forward_ms | routing_throughput_tokens_per_sec | peak_vram_bytes | bank_storage_bytes | optimizer_state_bytes | effective_support_count | dead_support_fraction | load_balance_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| approximate | bf16 | 2 | 0.68359 | 125.25 | 127.92 | 1.12e+07 | 6.4e+05 | 0 | 1.9705 | 0.99422 | 381.92 |
| approximate | fp16 | 3 | 0.41927 | 136.33 | 117.51 | 1.59e+07 | 1.82e+06 | 0 | 2.9262 | 0.99169 | 877.95 |
| approximate | fp32 | 3 | 0.39583 | 89.491 | 182.21 | 1.71e+07 | 5.57e+06 | 0 | 2.3561 | 0.92625 | 1904.8 |
| chunked | bf16 | 3 | 0.97526 | 111.48 | 144.32 | 1.99e+07 | 2.82e+06 | 0 | 3.0041 | 0.8713 | 1837.6 |
| chunked | fp16 | 3 | 1 | 134.07 | 121.28 | 1.99e+07 | 2.82e+06 | 0 | 1.9589 | 0.97993 | 1917 |
| chunked | fp32 | 2 | 1 | 84.744 | 189.04 | 1.53e+07 | 4.45e+06 | 0 | 3.7916 | 0.98814 | 687.13 |
| exact | bf16 | 2 | 0.97461 | 101.84 | 157.83 | 1.91e+07 | 2.61e+06 | 0 | 3.2881 | 0.99116 | 1056.4 |
| exact | fp16 | 3 | 0.99805 | 108.92 | 148.12 | 2.52e+07 | 4.13e+06 | 0 | 4.6999 | 0.99446 | 1440.9 |
| exact | fp32 | 3 | 1 | 79.427 | 201.99 | 1.31e+07 | 2.94e+06 | 0 | 2.621 | 0.85325 | 928.93 |
| product_key | bf16 | 3 | 0.96354 | 151.61 | 106.54 | 2.12e+07 | 9472 | 0 | 2.0736 | 0.99648 | 765.88 |
| product_key | fp16 | 2 | 0.98633 | 136.04 | 117.65 | 3.81e+07 | 16384 | 0 | 6.7681 | 0.99519 | 885.59 |
| product_key | fp32 | 3 | 1 | 77.139 | 209.75 | 1.17e+07 | 9514.7 | 0 | 2.447 | 0.98388 | 239.1 |

## Task/group summary

| router | benchmark_supports | n | recall_at_k_against_exact | routing_forward_ms | routing_throughput_tokens_per_sec | peak_vram_bytes | bank_storage_bytes | optimizer_state_bytes | effective_support_count | dead_support_fraction | load_balance_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| approximate | 1000 | 1 | 1 | 107.32 | 149.09 | 8.85e+06 | 1.28e+05 | 0 | 3.2964 | 0.78 | 28.315 |
| approximate | 16000 | 2 | 0.27344 | 132.44 | 120.85 | 1.26e+07 | 1.02e+06 | 0 | 3.415 | 0.98872 | 573.81 |
| approximate | 4000 | 2 | 1 | 125.91 | 127.3 | 9.71e+06 | 2.56e+05 | 0 | 1.252 | 0.994 | 211.06 |
| approximate | 64000 | 1 | 0.125 | 83.168 | 192.38 | 2.12e+07 | 8.19e+06 | 0 | 2.7719 | 0.999 | 1591 |
| approximate | 65536 | 2 | 0.070312 | 110.38 | 158.61 | 2.34e+07 | 6.29e+06 | 0 | 2.1927 | 0.99891 | 2961.5 |
| chunked | 1000 | 2 | 0.99805 | 130.76 | 123.6 | 8.79e+06 | 64000 | 0 | 2.8174 | 0.7795 | 31.702 |
| chunked | 4000 | 1 | 1 | 87.731 | 182.38 | 9.29e+06 | 5.12e+05 | 0 | 2.3148 | 0.984 | 130.9 |
| chunked | 65536 | 5 | 0.98594 | 111.37 | 149.06 | 2.47e+07 | 5.03e+06 | 0 | 2.9046 | 0.99739 | 2488.7 |
| exact | 1000 | 1 | 1 | 82.695 | 193.48 | 8.73e+06 | 1.28e+05 | 0 | 2.714 | 0.62 | 34.374 |
| exact | 16000 | 1 | 0.98828 | 108.75 | 147.13 | 1.26e+07 | 1.02e+06 | 0 | 3.2942 | 0.98425 | 434.13 |
| exact | 4000 | 1 | 1 | 73.725 | 217.02 | 9.3e+06 | 5.12e+05 | 0 | 3.5924 | 0.94025 | 92.902 |
| exact | 64000 | 3 | 0.99935 | 95.865 | 169.21 | 2.38e+07 | 5.46e+06 | 0 | 2.8967 | 0.99686 | 2044 |
| exact | 65536 | 2 | 0.97852 | 107.98 | 150.36 | 2.55e+07 | 4.19e+06 | 0 | 5.1241 | 0.99519 | 1264.4 |
| product_key | 1024 | 1 | 1 | 66.677 | 239.96 | 8.99e+06 | 4096 | 0 | 1 | 0.98438 | 63 |
| product_key | 16129 | 2 | 0.95312 | 106.07 | 161.68 | 1.58e+07 | 12192 | 0 | 1.7189 | 0.99802 | 602.76 |
| product_key | 4096 | 2 | 1 | 119.02 | 145.54 | 1.04e+07 | 6144 | 0 | 3.1406 | 0.98083 | 115.61 |
| product_key | 64009 | 1 | 0.98438 | 169.38 | 94.465 | 3.76e+07 | 16192 | 0 | 2.8427 | 0.999 | 1515.2 |
| product_key | 65536 | 2 | 0.98633 | 136.04 | 117.65 | 3.81e+07 | 16384 | 0 | 6.7681 | 0.99519 | 885.59 |

## Cross-check summary

| router | top_k | n | recall_at_k_against_exact | routing_forward_ms | routing_throughput_tokens_per_sec | peak_vram_bytes | bank_storage_bytes | optimizer_state_bytes | effective_support_count | dead_support_fraction | load_balance_error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| approximate | 1 | 2 | 0.53125 | 99.297 | 168.91 | 1.55e+07 | 4.32e+06 | 0 | 1 | 0.99788 | 2172 |
| approximate | 16 | 2 | 0.58984 | 121.16 | 133.81 | 1.07e+07 | 5.76e+05 | 0 | 3.5927 | 0.8825 | 330.55 |
| approximate | 2 | 1 | 1 | 131.21 | 121.94 | 9.71e+06 | 2.56e+05 | 0 | 1.504 | 0.992 | 173.12 |
| approximate | 4 | 1 | 0.125 | 83.168 | 192.38 | 2.12e+07 | 8.19e+06 | 0 | 2.7719 | 0.999 | 1591 |
| approximate | 8 | 2 | 0.22266 | 136.33 | 117.62 | 1.91e+07 | 2.61e+06 | 0 | 3.1633 | 0.99525 | 1171.4 |
| chunked | 1 | 2 | 0.96875 | 123.63 | 134.27 | 2.55e+07 | 4.19e+06 | 0 | 1 | 0.99976 | 4095 |
| chunked | 16 | 1 | 0.99219 | 116.61 | 137.21 | 2.55e+07 | 4.19e+06 | 0 | 4.4317 | 0.99614 | 1391 |
| chunked | 32 | 2 | 0.99805 | 99.724 | 165.83 | 1.51e+07 | 4.23e+06 | 0 | 4.4245 | 0.80514 | 635.13 |
| chunked | 4 | 3 | 1 | 114.27 | 145.82 | 1.45e+07 | 1.59e+06 | 0 | 2.3972 | 0.97467 | 595.59 |
| exact | 16 | 2 | 0.99414 | 91.235 | 182.08 | 1.1e+07 | 7.68e+05 | 0 | 3.4433 | 0.96225 | 263.51 |
| exact | 2 | 1 | 1 | 81.862 | 195.45 | 2.12e+07 | 8.19e+06 | 0 | 1.5565 | 0.9995 | 2659.5 |
| exact | 32 | 3 | 0.99805 | 100.17 | 163.66 | 1.98e+07 | 2.81e+06 | 0 | 4.77 | 0.86813 | 794.51 |
| exact | 4 | 1 | 1 | 108.93 | 146.88 | 2.51e+07 | 4.1e+06 | 0 | 2.5037 | 0.999 | 1973.5 |
| exact | 8 | 1 | 0.96094 | 94.942 | 168.52 | 2.55e+07 | 4.19e+06 | 0 | 3.282 | 0.99808 | 1678.7 |
| product_key | 1 | 1 | 1 | 66.677 | 239.96 | 8.99e+06 | 4096 | 0 | 1 | 0.98438 | 63 |
| product_key | 2 | 3 | 0.96875 | 121.35 | 142.89 | 1.4e+07 | 9493.3 | 0 | 1.7062 | 0.99615 | 458.23 |
| product_key | 32 | 1 | 0.99609 | 133.42 | 119.92 | 3.81e+07 | 16384 | 0 | 8.8874 | 0.99232 | 741.08 |
| product_key | 4 | 1 | 0.98438 | 169.38 | 94.465 | 3.76e+07 | 16192 | 0 | 2.8427 | 0.999 | 1515.2 |
| product_key | 8 | 2 | 0.98828 | 112.4 | 150.58 | 2.42e+07 | 12288 | 0 | 4.6246 | 0.98364 | 546.07 |

## Generated figures

- `memory_diagnostics.png`
- `router_load.png`

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
