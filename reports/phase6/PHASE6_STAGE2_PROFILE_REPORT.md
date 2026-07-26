# Phase 6 Stage2 Transformer Comparison report

- Row outputs: **48**
- Passing rows: **48**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage2_transformer_comparison/hpg_runs_profile_budget1`
- Manifest: `results/phase6/stage2_transformer_comparison/manifests/profile_hpg_38049074.jsonl`
- Artifact manifest: `results/phase6/stage2_transformer_comparison/hpg_runs_profile_budget1/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: controlled_symbolic_regimes=12, mqar=12, prototype=12, small_language=12
- `architecture`: T-KAM-ALT=6, T-KAM-F=5, T-KAM-L=5, T-KAM-VP=5, T-MEMTOK=5, T-MOE=5, T-PKM=6, T-WIDE=5, T0=6
- `scale`: 10M=16, 2M=16, 30M=16

## Primary grouped summary

| architecture | scale | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | trainable_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-KAM-ALT | 10M | 2 | 2.0085 | 0.441 | 1.565 | 4 | 256 | 1.01e+07 | 1.01e+07 | 7.9e+06 | 1e+07 | 0.0050343 | 32 | 32 | 4 | 1 | 3 | 14.92 |
| T-KAM-ALT | 30M | 4 | 3.8471 | 0.077632 | 1.0821 | 4 | 256 | 3e+07 | 3e+07 | 2.47e+07 | 3e+07 | 0.0013654 | 32 | 32 | 4 | 1 | 3 | 10.234 |
| T-KAM-F | 10M | 1 | 3.1314 | 0.67843 | 1.9708 | 4 | 256 | 1e+07 | 9.95e+06 | 9.9e+06 | 1e+07 | 0.000444 | 32 | 32 | 4 | 0 | 4 | 14.165 |
| T-KAM-F | 2M | 1 | 2.1089 | 1.2222 | 3.3948 | 4 | 256 | 1.99e+06 | 1.97e+06 | 1.95e+06 | 2e+06 | 0.003998 | 32 | 32 | 4 | 0 | 4 | 8.2545 |
| T-KAM-F | 30M | 3 | 3.8411 | 0.044964 | 1.0475 | 4 | 256 | 3e+07 | 2.99e+07 | 2.97e+07 | 3e+07 | 0.000507 | 32 | 32 | 4 | 0 | 4 | 17.322 |
| T-KAM-L | 10M | 4 | 3.7706 | 0.50033 | 1.776 | 4 | 256 | 9.96e+06 | 9.96e+06 | 8.83e+06 | 1e+07 | 0.0037757 | 32 | 32 | 4 | 0 | 4 | 7.5836 |
| T-KAM-L | 30M | 1 | 3.272 | 0.15725 | 1.1703 | 4 | 256 | 3e+07 | 3e+07 | 2.46e+07 | 3e+07 | 0.000719 | 32 | 32 | 4 | 0 | 4 | 6.7296 |
| T-KAM-VP | 10M | 1 | 2.0216 | 0.15697 | 1.17 | 4 | 256 | 1.01e+07 | 9.99e+06 | 7.9e+06 | 1e+07 | 0.0050343 | 32 | 32 | 4 | 0 | 4 | 15.177 |
| T-KAM-VP | 2M | 2 | 3.1516 | 1.3115 | 3.7291 | 4 | 256 | 1.99e+06 | 1.97e+06 | 1.69e+06 | 2e+06 | 0.002748 | 32 | 32 | 4 | 0 | 4 | 21.597 |
| T-KAM-VP | 30M | 2 | 2.5877 | 0.066925 | 1.0714 | 4 | 256 | 3e+07 | 2.99e+07 | 2.46e+07 | 3e+07 | 0.0010004 | 32 | 32 | 4 | 0 | 4 | 6.6299 |
| T-MEMTOK | 10M | 1 | 4.3879 | 0.060368 | 1.0622 | 4 | 256 | 1e+07 | 1e+07 | 9.97e+06 | 1e+07 | 0.000563 | 32 | 32 | 4 | 0 | 4 | 6.6189 |
| T-MEMTOK | 2M | 2 | 3.5888 | 1.3654 | 4.8258 | 4 | 256 | 2e+06 | 2e+06 | 1.98e+06 | 2e+06 | 0.0036 | 32 | 32 | 4 | 0 | 4 | 10.614 |
| T-MEMTOK | 30M | 2 | 3.9335 | 0.034789 | 1.036 | 4 | 256 | 3e+07 | 3e+07 | 2.99e+07 | 3e+07 | 0.000766 | 32 | 32 | 4 | 0 | 4 | 9.7739 |
| T-MOE | 10M | 3 | 3.0609 | 0.31856 | 1.3954 | 4 | 256 | 1e+07 | 1e+07 | 7.25e+06 | 1e+07 | 0.0014924 | 32 | 4 | 2 | 0 | 4 | 1.7101 |
| T-MOE | 2M | 2 | 2.618 | 1.2407 | 4.1327 | 4 | 256 | 2e+06 | 2e+06 | 1.46e+06 | 2e+06 | 0.002 | 32 | 4 | 2 | 0 | 4 | 3.8371 |
| T-PKM | 10M | 2 | 4.7303 | 0.11256 | 1.1194 | 4 | 256 | 1e+07 | 1e+07 | 9.97e+06 | 1e+07 | 0.000314 | 32 | 25 | 4 | 0 | 4 | 1.7886 |
| T-PKM | 2M | 4 | 3.1409 | 1.5681 | 5.1751 | 4 | 256 | 2e+06 | 2e+06 | 1.98e+06 | 2e+06 | 0.003212 | 32 | 25 | 4 | 0 | 4 | 2.072 |
| T-WIDE | 10M | 1 | 4.1222 | 0.065865 | 1.0681 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 1e+07 | 0.000208 | 32 | 0 | 4 | 0 | 4 | 0.78995 |
| T-WIDE | 2M | 2 | 2.8981 | 1.4374 | 4.3298 | 4 | 256 | 2e+06 | 2e+06 | 2e+06 | 2e+06 | 0.003648 | 32 | 0 | 4 | 0 | 4 | 1.094 |
| T-WIDE | 30M | 2 | 4.1334 | 0.014369 | 1.0145 | 4 | 256 | 3e+07 | 3e+07 | 3e+07 | 3e+07 | 0.000141 | 32 | 0 | 4 | 0 | 4 | 2.3456 |
| T0 | 10M | 1 | 4.3779 | 0.068036 | 1.0704 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 1e+07 | 0.000666 | 32 | 0 | 4 | 0 | 4 | 1.5402 |
| T0 | 2M | 3 | 3.4811 | 1.5526 | 4.8559 | 4 | 256 | 2e+06 | 2e+06 | 2e+06 | 2e+06 | 0.0038187 | 32 | 0 | 4 | 0 | 4 | 1.2269 |
| T0 | 30M | 2 | 4.1727 | 0.012316 | 1.0125 | 4 | 256 | 3e+07 | 3e+07 | 3e+07 | 3e+07 | 0.000915 | 32 | 0 | 4 | 0 | 4 | 1.2201 |

## Task/group summary

| task | architecture | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | trainable_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled_symbolic_regimes | T-KAM-ALT | 3 | 2.0391 | 0.33435 | 1.4196 | 4 | 256 | 1.67e+07 | 1.67e+07 | 1.35e+07 | 1.67e+07 | 0.0037835 | 32 | 32 | 4 | 1 | 3 | 12.242 |
| controlled_symbolic_regimes | T-KAM-F | 1 | 2.1089 | 1.2222 | 3.3948 | 4 | 256 | 1.99e+06 | 1.97e+06 | 1.95e+06 | 2e+06 | 0.003998 | 32 | 32 | 4 | 0 | 4 | 8.2545 |
| controlled_symbolic_regimes | T-KAM-VP | 3 | 2.0997 | 0.4583 | 1.8479 | 4 | 256 | 1.4e+07 | 1.39e+07 | 1.14e+07 | 1.4e+07 | 0.0039334 | 32 | 32 | 4 | 0 | 4 | 13.239 |
| controlled_symbolic_regimes | T-MOE | 2 | 2.1143 | 0.45146 | 1.5946 | 4 | 256 | 6e+06 | 6e+06 | 4.4e+06 | 6e+06 | 0.0013476 | 32 | 4 | 2 | 0 | 4 | 2.541 |
| controlled_symbolic_regimes | T-PKM | 1 | 2.2268 | 1.2955 | 3.6528 | 4 | 256 | 2e+06 | 2e+06 | 1.99e+06 | 2e+06 | 0.001728 | 32 | 25 | 4 | 0 | 4 | 1.7445 |
| controlled_symbolic_regimes | T-WIDE | 1 | 2.0369 | 1.1991 | 3.3173 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.99e+06 | 2e+06 | 0.005952 | 32 | 0 | 4 | 0 | 4 | 1.0674 |
| controlled_symbolic_regimes | T0 | 1 | 2.1278 | 1.2096 | 3.352 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.99e+06 | 2e+06 | 0.002624 | 32 | 0 | 4 | 0 | 4 | 1.06 |
| mqar | T-KAM-F | 2 | 4.2304 | 0.0067864 | 1.0068 | 4 | 256 | 3e+07 | 2.99e+07 | 2.98e+07 | 3e+07 | 0.000671 | 32 | 32 | 4 | 0 | 4 | 17.834 |
| mqar | T-KAM-L | 1 | 4.3739 | 0.13889 | 1.149 | 4 | 256 | 9.98e+06 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 6.5189 |
| mqar | T-MEMTOK | 2 | 4.5755 | 0.0308 | 1.0317 | 4 | 256 | 2e+07 | 2e+07 | 2e+07 | 2e+07 | 0.000899 | 32 | 32 | 4 | 0 | 4 | 6.8011 |
| mqar | T-WIDE | 2 | 4.003 | 0.84442 | 3.1778 | 4 | 256 | 1.6e+07 | 1.6e+07 | 1.6e+07 | 1.6e+07 | 0.000742 | 32 | 0 | 4 | 0 | 4 | 1.731 |
| mqar | T0 | 5 | 4.2078 | 0.7082 | 2.8622 | 4 | 256 | 1.48e+07 | 1.48e+07 | 1.48e+07 | 1.48e+07 | 0.0022656 | 32 | 0 | 4 | 0 | 4 | 1.3202 |
| prototype | T-KAM-ALT | 3 | 4.4294 | 0.06316 | 1.0666 | 4 | 256 | 3e+07 | 3e+07 | 2.47e+07 | 3e+07 | 0.0013932 | 32 | 32 | 4 | 1 | 3 | 11.351 |
| prototype | T-KAM-L | 1 | 4.4266 | 0.085278 | 1.089 | 4 | 256 | 9.98e+06 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 9.7411 |
| prototype | T-KAM-VP | 1 | 4.2397 | 1.4079 | 4.0874 | 4 | 256 | 2e+06 | 1.98e+06 | 1.7e+06 | 2e+06 | 1.2e-05 | 32 | 32 | 4 | 0 | 4 | 25.259 |
| prototype | T-MEMTOK | 1 | 4.1309 | 0.69698 | 2.0077 | 4 | 256 | 2.01e+06 | 2.01e+06 | 1.99e+06 | 2e+06 | 0.003922 | 32 | 32 | 4 | 0 | 4 | 12.607 |
| prototype | T-MOE | 1 | 3.9345 | 0.13491 | 1.1444 | 4 | 256 | 1e+07 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.589 |
| prototype | T-PKM | 3 | 4.6187 | 0.43284 | 1.7214 | 4 | 256 | 7.34e+06 | 7.34e+06 | 7.31e+06 | 7.33e+06 | 0.0013024 | 32 | 25 | 4 | 0 | 4 | 1.8877 |
| prototype | T-WIDE | 2 | 4.0712 | 0.04072 | 1.0419 | 4 | 256 | 2e+07 | 2e+07 | 2e+07 | 2e+07 | 0.000174 | 32 | 0 | 4 | 0 | 4 | 1.5699 |
| small_language | T-KAM-F | 2 | 3.0969 | 0.39987 | 1.5499 | 4 | 256 | 2e+07 | 1.99e+07 | 1.98e+07 | 2e+07 | 0.000312 | 32 | 32 | 4 | 0 | 4 | 15.231 |
| small_language | T-KAM-L | 3 | 3.1846 | 0.6448 | 2.0121 | 4 | 256 | 1.66e+07 | 1.66e+07 | 1.41e+07 | 1.67e+07 | 0.0037167 | 32 | 32 | 4 | 0 | 4 | 6.9347 |
| small_language | T-KAM-VP | 1 | 2.9615 | 0.13109 | 1.1401 | 4 | 256 | 3e+07 | 2.99e+07 | 2.46e+07 | 3e+07 | 0.000719 | 32 | 32 | 4 | 0 | 4 | 6.6566 |
| small_language | T-MEMTOK | 2 | 3.0753 | 1.0511 | 4.3574 | 4 | 256 | 1.6e+07 | 1.6e+07 | 1.59e+07 | 1.6e+07 | 0.0017872 | 32 | 32 | 4 | 0 | 4 | 10.593 |
| small_language | T-MOE | 2 | 3.1277 | 1.1997 | 4.059 | 4 | 256 | 6e+06 | 6e+06 | 4.41e+06 | 6e+06 | 0.0013092 | 32 | 4 | 2 | 0 | 4 | 2.5667 |
| small_language | T-PKM | 2 | 2.9707 | 1.9517 | 7.0611 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.98e+06 | 2e+06 | 0.00392 | 32 | 25 | 4 | 0 | 4 | 2.2288 |

## Cross-check summary

| architecture | task | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | trainable_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-KAM-ALT | controlled_symbolic_regimes | 3 | 2.0391 | 0.33435 | 1.4196 | 4 | 256 | 1.67e+07 | 1.67e+07 | 1.35e+07 | 1.67e+07 | 0.0037835 | 32 | 32 | 4 | 1 | 3 | 12.242 |
| T-KAM-ALT | prototype | 3 | 4.4294 | 0.06316 | 1.0666 | 4 | 256 | 3e+07 | 3e+07 | 2.47e+07 | 3e+07 | 0.0013932 | 32 | 32 | 4 | 1 | 3 | 11.351 |
| T-KAM-F | controlled_symbolic_regimes | 1 | 2.1089 | 1.2222 | 3.3948 | 4 | 256 | 1.99e+06 | 1.97e+06 | 1.95e+06 | 2e+06 | 0.003998 | 32 | 32 | 4 | 0 | 4 | 8.2545 |
| T-KAM-F | mqar | 2 | 4.2304 | 0.0067864 | 1.0068 | 4 | 256 | 3e+07 | 2.99e+07 | 2.98e+07 | 3e+07 | 0.000671 | 32 | 32 | 4 | 0 | 4 | 17.834 |
| T-KAM-F | small_language | 2 | 3.0969 | 0.39987 | 1.5499 | 4 | 256 | 2e+07 | 1.99e+07 | 1.98e+07 | 2e+07 | 0.000312 | 32 | 32 | 4 | 0 | 4 | 15.231 |
| T-KAM-L | mqar | 1 | 4.3739 | 0.13889 | 1.149 | 4 | 256 | 9.98e+06 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 6.5189 |
| T-KAM-L | prototype | 1 | 4.4266 | 0.085278 | 1.089 | 4 | 256 | 9.98e+06 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 9.7411 |
| T-KAM-L | small_language | 3 | 3.1846 | 0.6448 | 2.0121 | 4 | 256 | 1.66e+07 | 1.66e+07 | 1.41e+07 | 1.67e+07 | 0.0037167 | 32 | 32 | 4 | 0 | 4 | 6.9347 |
| T-KAM-VP | controlled_symbolic_regimes | 3 | 2.0997 | 0.4583 | 1.8479 | 4 | 256 | 1.4e+07 | 1.39e+07 | 1.14e+07 | 1.4e+07 | 0.0039334 | 32 | 32 | 4 | 0 | 4 | 13.239 |
| T-KAM-VP | prototype | 1 | 4.2397 | 1.4079 | 4.0874 | 4 | 256 | 2e+06 | 1.98e+06 | 1.7e+06 | 2e+06 | 1.2e-05 | 32 | 32 | 4 | 0 | 4 | 25.259 |
| T-KAM-VP | small_language | 1 | 2.9615 | 0.13109 | 1.1401 | 4 | 256 | 3e+07 | 2.99e+07 | 2.46e+07 | 3e+07 | 0.000719 | 32 | 32 | 4 | 0 | 4 | 6.6566 |
| T-MEMTOK | mqar | 2 | 4.5755 | 0.0308 | 1.0317 | 4 | 256 | 2e+07 | 2e+07 | 2e+07 | 2e+07 | 0.000899 | 32 | 32 | 4 | 0 | 4 | 6.8011 |
| T-MEMTOK | prototype | 1 | 4.1309 | 0.69698 | 2.0077 | 4 | 256 | 2.01e+06 | 2.01e+06 | 1.99e+06 | 2e+06 | 0.003922 | 32 | 32 | 4 | 0 | 4 | 12.607 |
| T-MEMTOK | small_language | 2 | 3.0753 | 1.0511 | 4.3574 | 4 | 256 | 1.6e+07 | 1.6e+07 | 1.59e+07 | 1.6e+07 | 0.0017872 | 32 | 32 | 4 | 0 | 4 | 10.593 |
| T-MOE | controlled_symbolic_regimes | 2 | 2.1143 | 0.45146 | 1.5946 | 4 | 256 | 6e+06 | 6e+06 | 4.4e+06 | 6e+06 | 0.0013476 | 32 | 4 | 2 | 0 | 4 | 2.541 |
| T-MOE | prototype | 1 | 3.9345 | 0.13491 | 1.1444 | 4 | 256 | 1e+07 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.589 |
| T-MOE | small_language | 2 | 3.1277 | 1.1997 | 4.059 | 4 | 256 | 6e+06 | 6e+06 | 4.41e+06 | 6e+06 | 0.0013092 | 32 | 4 | 2 | 0 | 4 | 2.5667 |
| T-PKM | controlled_symbolic_regimes | 1 | 2.2268 | 1.2955 | 3.6528 | 4 | 256 | 2e+06 | 2e+06 | 1.99e+06 | 2e+06 | 0.001728 | 32 | 25 | 4 | 0 | 4 | 1.7445 |
| T-PKM | prototype | 3 | 4.6187 | 0.43284 | 1.7214 | 4 | 256 | 7.34e+06 | 7.34e+06 | 7.31e+06 | 7.33e+06 | 0.0013024 | 32 | 25 | 4 | 0 | 4 | 1.8877 |
| T-PKM | small_language | 2 | 2.9707 | 1.9517 | 7.0611 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.98e+06 | 2e+06 | 0.00392 | 32 | 25 | 4 | 0 | 4 | 2.2288 |
| T-WIDE | controlled_symbolic_regimes | 1 | 2.0369 | 1.1991 | 3.3173 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.99e+06 | 2e+06 | 0.005952 | 32 | 0 | 4 | 0 | 4 | 1.0674 |
| T-WIDE | mqar | 2 | 4.003 | 0.84442 | 3.1778 | 4 | 256 | 1.6e+07 | 1.6e+07 | 1.6e+07 | 1.6e+07 | 0.000742 | 32 | 0 | 4 | 0 | 4 | 1.731 |
| T-WIDE | prototype | 2 | 4.0712 | 0.04072 | 1.0419 | 4 | 256 | 2e+07 | 2e+07 | 2e+07 | 2e+07 | 0.000174 | 32 | 0 | 4 | 0 | 4 | 1.5699 |
| T0 | controlled_symbolic_regimes | 1 | 2.1278 | 1.2096 | 3.352 | 4 | 256 | 1.99e+06 | 1.99e+06 | 1.99e+06 | 2e+06 | 0.002624 | 32 | 0 | 4 | 0 | 4 | 1.06 |
| T0 | mqar | 5 | 4.2078 | 0.7082 | 2.8622 | 4 | 256 | 1.48e+07 | 1.48e+07 | 1.48e+07 | 1.48e+07 | 0.0022656 | 32 | 0 | 4 | 0 | 4 | 1.3202 |

## Generated figures

- `learning_curves.png`
- `memory_diagnostics.png`
- `router_load.png`

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
