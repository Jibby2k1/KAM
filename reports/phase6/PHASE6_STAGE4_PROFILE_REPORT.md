# Phase 6 Stage4 Online Adaptation report

- Row outputs: **48**
- Passing rows: **48**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt2`
- Manifest: `results/phase6/stage4_online_adaptation/manifests/profile_hpg_38049769.jsonl`
- Artifact manifest: `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt2/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: mackey_glass_schedule=12, narma_schedule=12, prototype_schedule=12, symbolic_schedule=12
- `architecture`: T-KAM-DUAL=8, T-KAM-F=8, T-KAM-L=8, T-KAM-ONLINE=8, T-WIDE=8, T0=8
- `adapter`: episodic_insertion=6, expert_only=6, nlms=6, none=6, rls=6, sgd=6, slow_geometry=6, value_only=6

## Primary grouped summary

| architecture | adapter | n | global_nmse | early_nmse | late_nmse | reacquisition_time | geometry_update_count | memory_used | episodic_active |
|---|---|---|---|---|---|---|---|---|---|
| T-KAM-DUAL | expert_only | 1 | 1.7718 | 2.6688 | 0.84617 | 6 | 0 | 1 | 1 |
| T-KAM-DUAL | nlms | 1 | 0.65831 | 1.048 | 0.31697 | 0 | 0 | 1 | 1 |
| T-KAM-DUAL | rls | 1 | 0.23901 | 0.69818 | 0.0042275 | 98 | 0 | 1 | 1 |
| T-KAM-DUAL | sgd | 2 | 0.81872 | 0.9422 | 0.76121 | 0 | 0 | 1 | 1 |
| T-KAM-DUAL | slow_geometry | 1 | 1.4819 | 2.3381 | 0.55521 | 5 | 4 | 1 | 1 |
| T-KAM-DUAL | value_only | 2 | 0.98559 | 1.3447 | 0.79308 | 0 | 0 | 1 | 1 |
| T-KAM-F | episodic_insertion | 1 | 3.9136 | 3.6643 | 4.3613 | 0 | 0 | 1 | 1 |
| T-KAM-F | expert_only | 1 | 16.188 | 13.032 | 14.887 | 0 | 0 | 1 | 0 |
| T-KAM-F | nlms | 1 | 0.27219 | 0.79874 | 0.0056329 | 10 | 0 | 1 | 0 |
| T-KAM-F | none | 2 | 15.811 | 18.51 | 13.473 | 0 | 0 | 1 | 0 |
| T-KAM-F | slow_geometry | 3 | 5.3013 | 5.0256 | 4.3684 | 0 | 0 | 1 | 0 |
| T-KAM-L | episodic_insertion | 1 | 22.217 | 26.52 | 18.633 | 0 | 0 | 1 | 1 |
| T-KAM-L | expert_only | 2 | 8.7281 | 6.4798 | 8.7741 | 0 | 0 | 1 | 0 |
| T-KAM-L | nlms | 2 | 0.60081 | 0.80591 | 0.24203 | 5.5 | 0 | 1 | 0 |
| T-KAM-L | none | 2 | 24.992 | 26.757 | 23.488 | 14 | 0 | 1 | 0 |
| T-KAM-L | sgd | 1 | 1.8955 | 5.697 | 0.000512 | 112 | 0 | 1 | 0 |
| T-KAM-ONLINE | episodic_insertion | 1 | 57.697 | 61.118 | 53.476 | 27 | 0 | 1 | 1 |
| T-KAM-ONLINE | expert_only | 1 | 1.8571 | 5.462 | 0.044216 | 75 | 0 | 1 | 1 |
| T-KAM-ONLINE | none | 1 | 59.306 | 62.745 | 55.033 | 26 | 0 | 1 | 1 |
| T-KAM-ONLINE | rls | 2 | 0.46642 | 0.64531 | 0.3648 | 7.5 | 0 | 1 | 1 |
| T-KAM-ONLINE | sgd | 1 | 0.93347 | 1.5594 | 0.34134 | 0 | 0 | 1 | 1 |
| T-KAM-ONLINE | value_only | 2 | 1.1648 | 1.9084 | 0.6479 | 12 | 0 | 1 | 1 |
| T-WIDE | episodic_insertion | 1 | 3.3195 | 3.3185 | 3.3624 | 2 | 0 | 0 | 1 |
| T-WIDE | nlms | 2 | 0.58587 | 0.86113 | 0.40311 | 5 | 0 | 0 | 0 |
| T-WIDE | rls | 3 | 0.57895 | 0.8226 | 0.40264 | 20.667 | 0 | 0 | 0 |
| T-WIDE | slow_geometry | 1 | 1.981 | 5.95 | 0.0017619 | 108 | 0 | 0 | 0 |
| T-WIDE | value_only | 1 | 1.134 | 2.2684 | 0.35947 | 0 | 0 | 0 | 0 |
| T0 | episodic_insertion | 2 | 26.987 | 27.953 | 25.447 | 14.5 | 0 | 0 | 1 |
| T0 | expert_only | 1 | 1.0513 | 1.4533 | 0.84033 | 0 | 0 | 0 | 0 |
| T0 | none | 1 | 24.198 | 27.738 | 20.797 | 0 | 0 | 0 | 0 |
| T0 | sgd | 2 | 2.6673 | 2.4386 | 2.7744 | 0.5 | 0 | 0 | 0 |
| T0 | slow_geometry | 1 | 2.8733 | 2.3728 | 3.3723 | 0 | 0 | 0 | 0 |
| T0 | value_only | 1 | 0.95105 | 1.4127 | 0.76395 | 0 | 0 | 0 | 0 |

## Task/group summary

| stream_task | adapter | n | global_nmse | early_nmse | late_nmse | reacquisition_time | geometry_update_count | memory_used | episodic_active |
|---|---|---|---|---|---|---|---|---|---|
| mackey_glass_schedule | episodic_insertion | 2 | 54.157 | 56.792 | 50.474 | 27.5 | 0 | 0.5 | 1 |
| mackey_glass_schedule | expert_only | 1 | 1.8571 | 5.462 | 0.044216 | 75 | 0 | 1 | 1 |
| mackey_glass_schedule | nlms | 3 | 0.28174 | 0.83304 | 0.0035213 | 10.333 | 0 | 0.66667 | 0 |
| mackey_glass_schedule | none | 2 | 52.897 | 56.402 | 49.033 | 27 | 0 | 1 | 0.5 |
| mackey_glass_schedule | rls | 2 | 0.23381 | 0.69244 | 0.0021959 | 79 | 0 | 0.5 | 0.5 |
| mackey_glass_schedule | sgd | 1 | 1.8955 | 5.697 | 0.000512 | 112 | 0 | 1 | 0 |
| mackey_glass_schedule | slow_geometry | 1 | 1.981 | 5.95 | 0.0017619 | 108 | 0 | 0 | 0 |
| narma_schedule | episodic_insertion | 1 | 3.9136 | 3.6643 | 4.3613 | 0 | 0 | 1 | 1 |
| narma_schedule | expert_only | 1 | 1.0513 | 1.4533 | 0.84033 | 0 | 0 | 0 | 0 |
| narma_schedule | nlms | 1 | 0.89098 | 0.89126 | 0.80355 | 0 | 0 | 0 | 0 |
| narma_schedule | none | 1 | 3.4955 | 3.456 | 3.9443 | 0 | 0 | 1 | 0 |
| narma_schedule | rls | 1 | 0.76239 | 0.82552 | 0.71144 | 0 | 0 | 1 | 1 |
| narma_schedule | sgd | 3 | 0.90016 | 1.0705 | 0.79544 | 0 | 0 | 0.66667 | 0.66667 |
| narma_schedule | slow_geometry | 1 | 0.78535 | 1.0486 | 0.63163 | 0 | 0 | 1 | 0 |
| narma_schedule | value_only | 3 | 0.89849 | 1.1646 | 0.84741 | 0 | 0 | 0.66667 | 0.66667 |
| prototype_schedule | episodic_insertion | 1 | 22.217 | 26.52 | 18.633 | 0 | 0 | 1 | 1 |
| prototype_schedule | expert_only | 1 | 1.0272 | 1.5011 | 0.60264 | 0 | 0 | 1 | 0 |
| prototype_schedule | nlms | 2 | 0.78382 | 0.89523 | 0.39938 | 0 | 0 | 1 | 0.5 |
| prototype_schedule | none | 3 | 18.607 | 21.586 | 15.914 | 0 | 0 | 0.66667 | 0 |
| prototype_schedule | rls | 1 | 0.77939 | 0.99128 | 0.37635 | 0 | 0 | 0 | 0 |
| prototype_schedule | sgd | 1 | 0.93347 | 1.5594 | 0.34134 | 0 | 0 | 1 | 1 |
| prototype_schedule | slow_geometry | 1 | 0.96095 | 1.2048 | 0.72249 | 0 | 0 | 1 | 0 |
| prototype_schedule | value_only | 2 | 1.1196 | 1.972 | 0.53038 | 0 | 0 | 0.5 | 0.5 |
| symbolic_schedule | episodic_insertion | 2 | 3.3381 | 3.3787 | 3.3928 | 1.5 | 0 | 0 | 1 |
| symbolic_schedule | expert_only | 3 | 11.463 | 9.053 | 10.893 | 2 | 0 | 1 | 0.33333 |
| symbolic_schedule | rls | 2 | 0.44964 | 0.62746 | 0.42478 | 8.5 | 0 | 0.5 | 0.5 |
| symbolic_schedule | sgd | 1 | 4.2716 | 3.5503 | 4.6849 | 1 | 0 | 0 | 0 |
| symbolic_schedule | slow_geometry | 3 | 6.1709 | 5.8447 | 5.2262 | 1.6667 | 1.3333 | 0.66667 | 0.33333 |
| symbolic_schedule | value_only | 1 | 1.4511 | 2.7497 | 0.40238 | 24 | 0 | 1 | 1 |

## Cross-check summary

| architecture | stream_task | n | global_nmse | early_nmse | late_nmse | reacquisition_time | geometry_update_count | memory_used | episodic_active |
|---|---|---|---|---|---|---|---|---|---|
| T-KAM-DUAL | mackey_glass_schedule | 1 | 0.23901 | 0.69818 | 0.0042275 | 98 | 0 | 1 | 1 |
| T-KAM-DUAL | narma_schedule | 3 | 0.83444 | 0.9661 | 0.80243 | 0 | 0 | 1 | 1 |
| T-KAM-DUAL | prototype_schedule | 2 | 0.88181 | 1.3618 | 0.50913 | 0 | 0 | 1 | 1 |
| T-KAM-DUAL | symbolic_schedule | 2 | 1.6268 | 2.5034 | 0.70069 | 5.5 | 2 | 1 | 1 |
| T-KAM-F | mackey_glass_schedule | 1 | 0.27219 | 0.79874 | 0.0056329 | 10 | 0 | 1 | 0 |
| T-KAM-F | narma_schedule | 2 | 2.3495 | 2.3565 | 2.4965 | 0 | 0 | 1 | 0.5 |
| T-KAM-F | prototype_schedule | 3 | 10.861 | 12.742 | 9.2228 | 0 | 0 | 1 | 0 |
| T-KAM-F | symbolic_schedule | 2 | 15.173 | 12.927 | 13.319 | 0 | 0 | 1 | 0 |
| T-KAM-L | mackey_glass_schedule | 3 | 16.225 | 18.875 | 14.345 | 50.333 | 0 | 1 | 0 |
| T-KAM-L | narma_schedule | 1 | 3.4955 | 3.456 | 3.9443 | 0 | 0 | 1 | 0 |
| T-KAM-L | prototype_schedule | 3 | 8.0512 | 9.5879 | 6.5724 | 0 | 0 | 1 | 0.33333 |
| T-KAM-L | symbolic_schedule | 1 | 16.429 | 11.459 | 16.945 | 0 | 0 | 1 | 0 |
| T-KAM-ONLINE | mackey_glass_schedule | 3 | 39.62 | 43.108 | 36.185 | 42.667 | 0 | 1 | 1 |
| T-KAM-ONLINE | narma_schedule | 2 | 0.82046 | 0.94631 | 0.80243 | 0 | 0 | 1 | 1 |
| T-KAM-ONLINE | prototype_schedule | 1 | 0.93347 | 1.5594 | 0.34134 | 0 | 0 | 1 | 1 |
| T-KAM-ONLINE | symbolic_schedule | 2 | 0.8108 | 1.6074 | 0.21027 | 19.5 | 0 | 1 | 1 |
| T-WIDE | mackey_glass_schedule | 3 | 0.83012 | 2.4892 | 0.0015319 | 59.333 | 0 | 0 | 0 |
| T-WIDE | narma_schedule | 1 | 0.89098 | 0.89126 | 0.80355 | 0 | 0 | 0 | 0 |
| T-WIDE | prototype_schedule | 2 | 0.95668 | 1.6299 | 0.36791 | 0 | 0 | 0 | 0 |
| T-WIDE | symbolic_schedule | 2 | 2.0242 | 2.0541 | 2.0969 | 2 | 0 | 0 | 0.5 |
| T0 | mackey_glass_schedule | 1 | 50.616 | 52.467 | 47.471 | 28 | 0 | 0 | 1 |
| T0 | narma_schedule | 3 | 1.0218 | 1.3977 | 0.82273 | 0 | 0 | 0 | 0 |
| T0 | prototype_schedule | 1 | 24.198 | 27.738 | 20.797 | 0 | 0 | 0 | 0 |
| T0 | symbolic_schedule | 3 | 3.5005 | 3.1207 | 3.8268 | 0.66667 | 0 | 0 | 0.33333 |

## Generated figures

- `adaptation_curves.png`
- `memory_diagnostics.png`
- `router_load.png`

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
