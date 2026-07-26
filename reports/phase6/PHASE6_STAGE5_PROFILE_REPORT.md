# Phase 6 Stage5 Long Training report

- Row outputs: **12**
- Passing rows: **12**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage5_long_training/hpg_runs_profile_long2`
- Manifest: `results/phase6/stage5_long_training/manifests/profile_hpg_38050338.jsonl`
- Artifact manifest: `results/phase6/stage5_long_training/hpg_runs_profile_long2/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: prototype=4, small_language=4, switching_mackey_glass=4
- `architecture`: T-KAM-F=3, T-KAM-L=3, T-MOE=3, T0=3
- `scale`: 10M=6, 30M=6

## Primary grouped summary

| architecture | scale | n | initial_loss | loss | perplexity | training_steps | training_tokens | declared_token_budget | budget_completion_fraction | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-KAM-F | 10M | 2 | 4.4517 | 0.0046317 | 1.0046 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 23.386 |
| T-KAM-F | 30M | 1 | 4.3612 | 0.00066 | 1.0007 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 2.98e+07 | 3e+07 | 0.000671 | 32 | 32 | 4 | 30.137 |
| T-KAM-L | 10M | 1 | 3.1398 | 0.0011465 | 1.0011 | 64 | 4096 | 6e+08 | 6.83e-06 | 9.95e+06 | 8.81e+06 | 1e+07 | 0.0052157 | 32 | 32 | 4 | 7.423 |
| T-KAM-L | 30M | 2 | 4.0402 | 5.34e-05 | 1.0001 | 64 | 4096 | 1.1e+09 | 1.13e-05 | 3e+07 | 2.47e+07 | 3e+07 | 0.001056 | 32 | 32 | 4 | 9.0615 |
| T-MOE | 10M | 2 | 3.6529 | 0.0044942 | 1.0045 | 64 | 4096 | 1.1e+09 | 1.13e-05 | 1e+07 | 7.21e+06 | 1e+07 | 0.002083 | 32 | 4 | 2 | 2.1773 |
| T-MOE | 30M | 1 | 4.2991 | 0.0021629 | 1.0022 | 64 | 4096 | 6e+08 | 6.83e-06 | 3e+07 | 1.94e+07 | 3e+07 | 2.35e-05 | 32 | 4 | 2 | 3.7593 |
| T0 | 10M | 1 | 4.0976 | 0.0012177 | 1.0012 | 64 | 4096 | 6e+08 | 6.83e-06 | 1e+07 | 1e+07 | 1e+07 | 0.000666 | 32 | 0 | 4 | 1.6244 |
| T0 | 30M | 2 | 3.8541 | 0.011018 | 1.0111 | 64 | 4096 | 1.1e+09 | 1.13e-05 | 3e+07 | 3e+07 | 3e+07 | 0.000635 | 32 | 0 | 4 | 1.7007 |

## Task/group summary

| task | architecture | n | initial_loss | loss | perplexity | training_steps | training_tokens | declared_token_budget | budget_completion_fraction | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prototype | T-KAM-F | 2 | 4.4517 | 0.0046317 | 1.0046 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 23.386 |
| prototype | T-KAM-L | 1 | 4.8988 | 2.99e-05 | 1 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 2.47e+07 | 3e+07 | 0.0013932 | 32 | 32 | 4 | 11.404 |
| prototype | T-MOE | 1 | 4.2042 | 0.0076365 | 1.0077 | 64 | 4096 | 2e+08 | 2.05e-05 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 2.6366 |
| small_language | T-KAM-L | 2 | 3.1607 | 0.000612 | 1.0006 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 2e+07 | 1.67e+07 | 2e+07 | 0.0029673 | 32 | 32 | 4 | 7.0709 |
| small_language | T-MOE | 1 | 3.1016 | 0.0013518 | 1.0014 | 64 | 4096 | 2e+09 | 2.05e-06 | 1e+07 | 7.35e+06 | 1e+07 | 0.0010024 | 32 | 4 | 2 | 1.718 |
| small_language | T0 | 1 | 3.1715 | 0.021974 | 1.0222 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 3e+07 | 3e+07 | 0.000354 | 32 | 0 | 4 | 2.0008 |
| switching_mackey_glass | T-KAM-F | 1 | 4.3612 | 0.00066 | 1.0007 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 2.98e+07 | 3e+07 | 0.000671 | 32 | 32 | 4 | 30.137 |
| switching_mackey_glass | T-MOE | 1 | 4.2991 | 0.0021629 | 1.0022 | 64 | 4096 | 6e+08 | 6.83e-06 | 3e+07 | 1.94e+07 | 3e+07 | 2.35e-05 | 32 | 4 | 2 | 3.7593 |
| switching_mackey_glass | T0 | 2 | 4.3172 | 0.00064 | 1.0006 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 2e+07 | 2e+07 | 2e+07 | 0.00079 | 32 | 0 | 4 | 1.5125 |

## Cross-check summary

| architecture | task | n | initial_loss | loss | perplexity | training_steps | training_tokens | declared_token_budget | budget_completion_fraction | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-KAM-F | prototype | 2 | 4.4517 | 0.0046317 | 1.0046 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 23.386 |
| T-KAM-F | switching_mackey_glass | 1 | 4.3612 | 0.00066 | 1.0007 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 2.98e+07 | 3e+07 | 0.000671 | 32 | 32 | 4 | 30.137 |
| T-KAM-L | prototype | 1 | 4.8988 | 2.99e-05 | 1 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 2.47e+07 | 3e+07 | 0.0013932 | 32 | 32 | 4 | 11.404 |
| T-KAM-L | small_language | 2 | 3.1607 | 0.000612 | 1.0006 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 2e+07 | 1.67e+07 | 2e+07 | 0.0029673 | 32 | 32 | 4 | 7.0709 |
| T-MOE | prototype | 1 | 4.2042 | 0.0076365 | 1.0077 | 64 | 4096 | 2e+08 | 2.05e-05 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 2.6366 |
| T-MOE | small_language | 1 | 3.1016 | 0.0013518 | 1.0014 | 64 | 4096 | 2e+09 | 2.05e-06 | 1e+07 | 7.35e+06 | 1e+07 | 0.0010024 | 32 | 4 | 2 | 1.718 |
| T-MOE | switching_mackey_glass | 1 | 4.2991 | 0.0021629 | 1.0022 | 64 | 4096 | 6e+08 | 6.83e-06 | 3e+07 | 1.94e+07 | 3e+07 | 2.35e-05 | 32 | 4 | 2 | 3.7593 |
| T0 | small_language | 1 | 3.1715 | 0.021974 | 1.0222 | 64 | 4096 | 2e+08 | 2.05e-05 | 3e+07 | 3e+07 | 3e+07 | 0.000354 | 32 | 0 | 4 | 2.0008 |
| T0 | switching_mackey_glass | 2 | 4.3172 | 0.00064 | 1.0006 | 64 | 4096 | 1.3e+09 | 4.44e-06 | 2e+07 | 2e+07 | 2e+07 | 0.00079 | 32 | 0 | 4 | 1.5125 |

## Generated figures

- `learning_curves.png`
- `memory_diagnostics.png`
- `router_load.png`

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
