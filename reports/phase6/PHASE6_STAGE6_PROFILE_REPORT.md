# Phase 6 Stage6 Confirmation report

- Row outputs: **12**
- Passing rows: **12**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage6_confirmation/hpg_runs_profile_confirm1`
- Manifest: `results/phase6/stage6_confirmation/manifests/profile_hpg_38050441.jsonl`
- Artifact manifest: `results/phase6/stage6_confirmation/hpg_runs_profile_confirm1/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: mqar=3, prototype=3, small_language=3, switching_mackey_glass=3
- `architecture`: T-KAM-ALT=1, T-KAM-F=1, T-KAM-L=2, T-KAM-VP=2, T-MOE=2, T-PKM=1, T-WIDE=1, T0=2
- `scale`: 10M=12
- `claim`: alternating_vs_joint=4, kam_vs_moe_pkm=4, kam_vs_widened_ffn=4

## Primary grouped summary

| claim | architecture | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alternating_vs_joint | T-KAM-F | 1 | 4.7077 | 0.66528 | 1.945 | 4 | 256 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 0 | 4 | 21.824 |
| alternating_vs_joint | T-MOE | 1 | 4.8082 | 0.42489 | 1.5294 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.7662 |
| alternating_vs_joint | T-PKM | 1 | 3.1855 | 0.68487 | 1.9835 | 4 | 256 | 1e+07 | 9.98e+06 | 1e+07 | 0.0027776 | 32 | 25 | 4 | 0 | 4 | 2.9279 |
| alternating_vs_joint | T0 | 1 | 4.6036 | 0.070168 | 1.0727 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000666 | 32 | 0 | 4 | 0 | 4 | 1.7341 |
| kam_vs_moe_pkm | T-KAM-ALT | 1 | 4.5956 | 0.35855 | 1.4313 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 1 | 3 | 10.047 |
| kam_vs_moe_pkm | T-KAM-VP | 1 | 4.2789 | 0.56439 | 1.7584 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 10.48 |
| kam_vs_moe_pkm | T-MOE | 1 | 4.0488 | 0.10193 | 1.1073 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.9512 |
| kam_vs_moe_pkm | T-WIDE | 1 | 4.1049 | 0.077663 | 1.0808 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000208 | 32 | 0 | 4 | 0 | 4 | 0.88508 |
| kam_vs_widened_ffn | T-KAM-L | 2 | 3.9637 | 0.14725 | 1.1586 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 11.134 |
| kam_vs_widened_ffn | T-KAM-VP | 1 | 3.1683 | 0.83694 | 2.3093 | 4 | 256 | 9.95e+06 | 8.81e+06 | 1e+07 | 0.0052157 | 32 | 32 | 4 | 0 | 4 | 7.2445 |
| kam_vs_widened_ffn | T0 | 1 | 3.0643 | 0.56513 | 1.7597 | 4 | 256 | 9.98e+06 | 9.98e+06 | 1e+07 | 0.0020448 | 32 | 0 | 4 | 0 | 4 | 2.1866 |

## Task/group summary

| task | architecture | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mqar | T-MOE | 1 | 4.0488 | 0.10193 | 1.1073 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.9512 |
| mqar | T-WIDE | 1 | 4.1049 | 0.077663 | 1.0808 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000208 | 32 | 0 | 4 | 0 | 4 | 0.88508 |
| mqar | T0 | 1 | 4.6036 | 0.070168 | 1.0727 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000666 | 32 | 0 | 4 | 0 | 4 | 1.7341 |
| prototype | T-KAM-ALT | 1 | 4.5956 | 0.35855 | 1.4313 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 1 | 3 | 10.047 |
| prototype | T-KAM-L | 2 | 3.9637 | 0.14725 | 1.1586 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 11.134 |
| small_language | T-KAM-VP | 1 | 3.1683 | 0.83694 | 2.3093 | 4 | 256 | 9.95e+06 | 8.81e+06 | 1e+07 | 0.0052157 | 32 | 32 | 4 | 0 | 4 | 7.2445 |
| small_language | T-PKM | 1 | 3.1855 | 0.68487 | 1.9835 | 4 | 256 | 1e+07 | 9.98e+06 | 1e+07 | 0.0027776 | 32 | 25 | 4 | 0 | 4 | 2.9279 |
| small_language | T0 | 1 | 3.0643 | 0.56513 | 1.7597 | 4 | 256 | 9.98e+06 | 9.98e+06 | 1e+07 | 0.0020448 | 32 | 0 | 4 | 0 | 4 | 2.1866 |
| switching_mackey_glass | T-KAM-F | 1 | 4.7077 | 0.66528 | 1.945 | 4 | 256 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 0 | 4 | 21.824 |
| switching_mackey_glass | T-KAM-VP | 1 | 4.2789 | 0.56439 | 1.7584 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 10.48 |
| switching_mackey_glass | T-MOE | 1 | 4.8082 | 0.42489 | 1.5294 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.7662 |

## Cross-check summary

| architecture | task | n | initial_loss | loss | perplexity | training_steps | training_tokens | total_parameters | active_parameter_count | target_parameter_budget | parameter_match_error_fraction | declared_memory_slots | effective_memory_slots | effective_top_k | geometry_update_steps | algebra_update_steps | measured_forward_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-KAM-ALT | prototype | 1 | 4.5956 | 0.35855 | 1.4313 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 1 | 3 | 10.047 |
| T-KAM-F | switching_mackey_glass | 1 | 4.7077 | 0.66528 | 1.945 | 4 | 256 | 1e+07 | 9.92e+06 | 1e+07 | 0.0015719 | 32 | 32 | 4 | 0 | 4 | 21.824 |
| T-KAM-L | prototype | 2 | 3.9637 | 0.14725 | 1.1586 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 11.134 |
| T-KAM-VP | small_language | 1 | 3.1683 | 0.83694 | 2.3093 | 4 | 256 | 9.95e+06 | 8.81e+06 | 1e+07 | 0.0052157 | 32 | 32 | 4 | 0 | 4 | 7.2445 |
| T-KAM-VP | switching_mackey_glass | 1 | 4.2789 | 0.56439 | 1.7584 | 4 | 256 | 9.98e+06 | 8.84e+06 | 1e+07 | 0.0023357 | 32 | 32 | 4 | 0 | 4 | 10.48 |
| T-MOE | mqar | 1 | 4.0488 | 0.10193 | 1.1073 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.9512 |
| T-MOE | switching_mackey_glass | 1 | 4.8082 | 0.42489 | 1.5294 | 4 | 256 | 1e+07 | 7.07e+06 | 1e+07 | 0.0031636 | 32 | 4 | 2 | 0 | 4 | 2.7662 |
| T-PKM | small_language | 1 | 3.1855 | 0.68487 | 1.9835 | 4 | 256 | 1e+07 | 9.98e+06 | 1e+07 | 0.0027776 | 32 | 25 | 4 | 0 | 4 | 2.9279 |
| T-WIDE | mqar | 1 | 4.1049 | 0.077663 | 1.0808 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000208 | 32 | 0 | 4 | 0 | 4 | 0.88508 |
| T0 | mqar | 1 | 4.6036 | 0.070168 | 1.0727 | 4 | 256 | 1e+07 | 1e+07 | 1e+07 | 0.000666 | 32 | 0 | 4 | 0 | 4 | 1.7341 |
| T0 | small_language | 1 | 3.0643 | 0.56513 | 1.7597 | 4 | 256 | 9.98e+06 | 9.98e+06 | 1e+07 | 0.0020448 | 32 | 0 | 4 | 0 | 4 | 2.1866 |

## Generated figures

- `learning_curves.png`
- `memory_diagnostics.png`
- `router_load.png`

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
