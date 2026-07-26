# Phase 6 Stage1 Mechanism report

- Row outputs: **3000**
- Passing rows: **3000**
- Failed/non-passing rows: **0**
- Row-level execution gate: **PASS**
- Run root: `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3`
- Manifest: `results/phase6/stage1_mechanism/manifests/full.jsonl`
- Artifact manifest: `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/artifact_manifest.json`

This is a descriptive screen. It does not establish a paired treatment effect, a scaling law, or a promotion decision; those require the declared seed-paired and held-out analyses.

## Factor coverage

- `task`: mqar=750, prototype=750, switching_mackey_glass=750, switching_narma=750
- `architecture`: T-KAM-F=600, T-KAM-L=600, T-MEMTOK=600, T-WIDE=600, T0=600
- `optimizer`: alternating_128_1=375, alternating_32_1=375, alternating_8_1=375, dictionary_update=375, joint_sgd=375, ridge_resolve=375, variable_projection_implicit=375, variable_projection_stopgrad=375
- `geometry`: fixed_data_sample=500, fixed_farthest_point=500, fixed_kmeans=500, fixed_random=500, learned_full=500, learned_low_rank_delta=500

## Primary grouped summary

| task | architecture | n | initial_loss | loss | training_steps | alternating_geometry_steps | solver_condition_number | measured_forward_ms | peak_vram_bytes |
|---|---|---|---|---|---|---|---|---|---|
| mqar | T-KAM-F | 161 | 0.025874 | 0.012296 | 2.5776 | 0.30159 | 52775 | 4.0651 | 1.56e+07 |
| mqar | T-KAM-L | 138 | 0.024224 | 0.0093603 | 4.1667 | 0.42593 | 51626 | 4.0468 | 1.6e+07 |
| mqar | T-MEMTOK | 148 | 0.02496 | 0.013791 | 1.4797 | 0 | 52011 | 4.0993 | 1.52e+07 |
| mqar | T-WIDE | 159 | 0.017359 | 0.0080944 | 1 | 0 | 48866 | 0.050368 | 1.54e+07 |
| mqar | T0 | 144 | 0.023693 | 0.012255 | 1 | 0 | 51601 | 0.033161 | 1.47e+07 |
| prototype | T-KAM-F | 150 | 0.010589 | 0.0034733 | 2.66 | 0.33333 | 9.63e+05 | 4.0951 | 1.54e+07 |
| prototype | T-KAM-L | 156 | 0.010737 | 0.0030166 | 4.1282 | 0.35714 | 9.63e+05 | 4.0952 | 1.59e+07 |
| prototype | T-MEMTOK | 147 | 0.010585 | 0.00493 | 1.4898 | 0 | 9.63e+05 | 4.0505 | 1.5e+07 |
| prototype | T-WIDE | 140 | 0.0061874 | 0.0012994 | 1 | 0 | 1.15e+06 | 0.049523 | 1.47e+07 |
| prototype | T0 | 157 | 0.010633 | 0.0055284 | 1 | 0 | 9.63e+05 | 0.032712 | 1.5e+07 |
| switching_mackey_glass | T-KAM-F | 146 | 0.037472 | 0.016566 | 2.6301 | 0.2963 | 1.57e+06 | 4.1273 | 1.59e+07 |
| switching_mackey_glass | T-KAM-L | 150 | 0.037424 | 0.0155 | 4.1067 | 0.30769 | 1.57e+06 | 4.1263 | 1.76e+07 |
| switching_mackey_glass | T-MEMTOK | 138 | 0.037464 | 0.018626 | 1.5145 | 0 | 1.57e+06 | 3.9948 | 1.47e+07 |
| switching_mackey_glass | T-WIDE | 164 | 0.027302 | 0.012661 | 1 | 0 | 1.25e+06 | 0.04957 | 1.52e+07 |
| switching_mackey_glass | T0 | 152 | 0.037507 | 0.019822 | 1 | 0 | 1.57e+06 | 0.032812 | 1.48e+07 |
| switching_narma | T-KAM-F | 143 | 0.039187 | 0.019711 | 2.6084 | 0.31481 | 1.63e+06 | 4.1389 | 1.71e+07 |
| switching_narma | T-KAM-L | 156 | 0.037138 | 0.013603 | 4.1282 | 0.3125 | 1.64e+06 | 4.1212 | 1.78e+07 |
| switching_narma | T-MEMTOK | 167 | 0.037972 | 0.019959 | 1.515 | 0 | 1.63e+06 | 4.022 | 1.51e+07 |
| switching_narma | T-WIDE | 137 | 0.024287 | 0.011083 | 1 | 0 | 1.25e+06 | 0.049356 | 1.52e+07 |
| switching_narma | T0 | 147 | 0.036628 | 0.020752 | 1 | 0 | 1.63e+06 | 0.032945 | 1.52e+07 |

## Task/group summary

| task | optimizer | n | initial_loss | loss | training_steps | alternating_geometry_steps | solver_condition_number | measured_forward_ms | peak_vram_bytes |
|---|---|---|---|---|---|---|---|---|---|
| mqar | alternating_128_1 | 101 | 0.022231 | 0.016433 | 2.0594 | 0.14851 | — | 2.1998 | 1.83e+07 |
| mqar | alternating_32_1 | 80 | 0.023254 | 0.01678 | 2.1125 | 0.175 | — | 2.2785 | 1.83e+07 |
| mqar | alternating_8_1 | 105 | 0.023556 | 0.017375 | 2.181 | 0.12381 | — | 2.763 | 1.83e+07 |
| mqar | dictionary_update | 90 | 0.022995 | 0.016819 | 2.0556 | — | — | 2.4606 | 1.83e+07 |
| mqar | joint_sgd | 95 | 0.022857 | 0.017125 | 1.8842 | — | — | 2.4508 | 1.83e+07 |
| mqar | ridge_resolve | 94 | 0.0231 | 0.0015742 | 1.8617 | — | 51435 | 2.4394 | 9.72e+06 |
| mqar | variable_projection_implicit | 102 | 0.023822 | 0.0012546 | 2.049 | — | 51580 | 2.5271 | 1.11e+07 |
| mqar | variable_projection_stopgrad | 83 | 0.023532 | 0.0014588 | 1.9036 | — | 51222 | 2.3674 | 1.09e+07 |
| prototype | alternating_128_1 | 93 | 0.0097173 | 0.0060648 | 2.043 | 0.096774 | — | 2.3485 | 1.82e+07 |
| prototype | alternating_32_1 | 99 | 0.0097569 | 0.0061954 | 2.1717 | 0.13131 | — | 2.7297 | 1.83e+07 |
| prototype | alternating_8_1 | 88 | 0.0098614 | 0.0061259 | 2.1932 | 0.19318 | — | 2.4632 | 1.83e+07 |
| prototype | dictionary_update | 86 | 0.0096843 | 0.0059616 | 2.0581 | — | — | 2.1423 | 1.82e+07 |
| prototype | joint_sgd | 87 | 0.0099362 | 0.0061703 | 2.1494 | — | — | 2.6608 | 1.82e+07 |
| prototype | ridge_resolve | 95 | 0.010148 | 6.19e-07 | 2.1684 | — | 9.9e+05 | 2.7358 | 9.71e+06 |
| prototype | variable_projection_implicit | 104 | 0.0094779 | 7.53e-07 | 1.9808 | — | 1.01e+06 | 2.3582 | 1.13e+07 |
| prototype | variable_projection_stopgrad | 98 | 0.0099021 | 7.26e-07 | 1.8878 | — | 1e+06 | 2.3912 | 1.07e+07 |
| switching_mackey_glass | alternating_128_1 | 87 | 0.035882 | 0.027734 | 1.954 | 0.10345 | — | 2.2322 | 1.86e+07 |
| switching_mackey_glass | alternating_32_1 | 94 | 0.034812 | 0.025867 | 2.2553 | 0.12766 | — | 2.5801 | 1.88e+07 |
| switching_mackey_glass | alternating_8_1 | 95 | 0.035588 | 0.027087 | 1.9895 | 0.11579 | — | 2.426 | 1.86e+07 |
| switching_mackey_glass | dictionary_update | 90 | 0.035331 | 0.026484 | 2.1 | — | — | 2.3265 | 1.87e+07 |
| switching_mackey_glass | joint_sgd | 97 | 0.035609 | 0.026828 | 2.0515 | — | — | 2.3115 | 1.88e+07 |
| switching_mackey_glass | ridge_resolve | 111 | 0.035579 | 1.12e-07 | 2.1351 | — | 1.51e+06 | 2.7271 | 1.02e+07 |
| switching_mackey_glass | variable_projection_implicit | 81 | 0.034616 | 1.13e-07 | 1.7284 | — | 1.5e+06 | 1.821 | 1.11e+07 |
| switching_mackey_glass | variable_projection_stopgrad | 95 | 0.034436 | 1.1e-07 | 1.9895 | — | 1.5e+06 | 2.4728 | 1.11e+07 |
| switching_narma | alternating_128_1 | 94 | 0.035831 | 0.025791 | 2.2234 | 0.11702 | — | 2.3569 | 1.88e+07 |
| switching_narma | alternating_32_1 | 102 | 0.035617 | 0.025884 | 2.2255 | 0.14706 | — | 2.6424 | 1.88e+07 |
| switching_narma | alternating_8_1 | 87 | 0.034708 | 0.025188 | 2.1494 | 0.12644 | — | 2.5906 | 1.88e+07 |
| switching_narma | dictionary_update | 109 | 0.034385 | 0.024983 | 2 | — | — | 2.7054 | 1.87e+07 |
| switching_narma | joint_sgd | 96 | 0.035153 | 0.025887 | 1.9479 | — | — | 2.5369 | 1.87e+07 |
| switching_narma | ridge_resolve | 75 | 0.035177 | 0.0014067 | 1.9733 | — | 1.57e+06 | 2.4793 | 1.03e+07 |
| switching_narma | variable_projection_implicit | 88 | 0.037307 | 0.0014823 | 2.0114 | — | 1.57e+06 | 2.7272 | 1.11e+07 |
| switching_narma | variable_projection_stopgrad | 99 | 0.0342 | 0.0014424 | 2.0303 | — | 1.54e+06 | 2.3962 | 1.16e+07 |

## Cross-check summary

| architecture | geometry | n | initial_loss | loss | training_steps | alternating_geometry_steps | solver_condition_number | measured_forward_ms | peak_vram_bytes |
|---|---|---|---|---|---|---|---|---|---|
| T-KAM-F | fixed_data_sample | 104 | 0.027857 | 0.012651 | 2.4423 | 0 | 9.35e+05 | 4.0906 | 1.52e+07 |
| T-KAM-F | fixed_farthest_point | 103 | 0.02802 | 0.013263 | 2.5049 | 0 | 9.66e+05 | 4.115 | 1.54e+07 |
| T-KAM-F | fixed_kmeans | 96 | 0.026885 | 0.012178 | 2.3958 | 0 | 9.88e+05 | 4.1116 | 1.48e+07 |
| T-KAM-F | fixed_random | 103 | 0.029372 | 0.013067 | 2.6602 | 0 | 1.07e+06 | 4.1034 | 1.52e+07 |
| T-KAM-F | learned_full | 97 | 0.027067 | 0.012538 | 2.8866 | 1 | 9.8e+05 | 4.1151 | 1.76e+07 |
| T-KAM-F | learned_low_rank_delta | 97 | 0.029008 | 0.01366 | 2.8351 | 1 | 1.07e+06 | 4.097 | 1.78e+07 |
| T-KAM-L | fixed_data_sample | 97 | 0.028531 | 0.011172 | 4 | 0 | 1.06e+06 | 4.0984 | 1.6e+07 |
| T-KAM-L | fixed_farthest_point | 113 | 0.027219 | 0.010565 | 4 | 0 | 1.1e+06 | 4.1138 | 1.63e+07 |
| T-KAM-L | fixed_kmeans | 90 | 0.027573 | 0.010395 | 4 | 0 | 1.19e+06 | 4.1208 | 1.61e+07 |
| T-KAM-L | fixed_random | 100 | 0.026315 | 0.010167 | 4 | 0 | 8.23e+05 | 4.1074 | 1.59e+07 |
| T-KAM-L | learned_full | 94 | 0.027482 | 0.010596 | 4.4362 | 1 | 8.33e+05 | 4.0679 | 1.84e+07 |
| T-KAM-L | learned_low_rank_delta | 106 | 0.02722 | 0.0092798 | 4.3585 | 1 | 1.27e+06 | 4.0826 | 1.81e+07 |
| T-MEMTOK | fixed_data_sample | 97 | 0.026435 | 0.014309 | 1.3814 | 0 | 1.14e+06 | 4.0642 | 1.51e+07 |
| T-MEMTOK | fixed_farthest_point | 95 | 0.027042 | 0.015065 | 1.5263 | 0 | 1.06e+06 | 4.045 | 1.55e+07 |
| T-MEMTOK | fixed_kmeans | 105 | 0.028507 | 0.014789 | 1.581 | 0 | 9.47e+05 | 4.0186 | 1.5e+07 |
| T-MEMTOK | fixed_random | 93 | 0.025234 | 0.011807 | 1.5054 | 0 | 1.1e+06 | 4.0376 | 1.46e+07 |
| T-MEMTOK | learned_full | 112 | 0.029453 | 0.015167 | 1.4911 | 0 | 1.09e+06 | 4.0438 | 1.5e+07 |
| T-MEMTOK | learned_low_rank_delta | 98 | 0.030508 | 0.015313 | 1.5102 | 0 | 1.18e+06 | 4.0429 | 1.5e+07 |
| T-WIDE | fixed_data_sample | 100 | 0.020004 | 0.0088349 | 1 | 0 | 8.68e+05 | 0.049093 | 1.52e+07 |
| T-WIDE | fixed_farthest_point | 95 | 0.019556 | 0.0087341 | 1 | 0 | 8.57e+05 | 0.049639 | 1.51e+07 |
| T-WIDE | fixed_kmeans | 109 | 0.019088 | 0.0095965 | 1 | 0 | 9.58e+05 | 0.049644 | 1.56e+07 |
| T-WIDE | fixed_random | 110 | 0.01817 | 0.0078314 | 1 | 0 | 1.12e+06 | 0.050019 | 1.53e+07 |
| T-WIDE | learned_full | 94 | 0.018271 | 0.0077403 | 1 | 0 | 8.12e+05 | 0.050284 | 1.51e+07 |
| T-WIDE | learned_low_rank_delta | 92 | 0.019307 | 0.0077769 | 1 | 0 | 9.6e+05 | 0.049651 | 1.45e+07 |
| T0 | fixed_data_sample | 102 | 0.028441 | 0.014809 | 1 | 0 | 1.01e+06 | 0.032915 | 1.47e+07 |
| T0 | fixed_farthest_point | 94 | 0.025372 | 0.012944 | 1 | 0 | 9.78e+05 | 0.03271 | 1.46e+07 |
| T0 | fixed_kmeans | 100 | 0.025165 | 0.014062 | 1 | 0 | 9.53e+05 | 0.033106 | 1.51e+07 |
| T0 | fixed_random | 94 | 0.027756 | 0.016213 | 1 | 0 | 1.12e+06 | 0.033139 | 1.53e+07 |
| T0 | learned_full | 103 | 0.02642 | 0.015011 | 1 | 0 | 1.17e+06 | 0.032738 | 1.55e+07 |
| T0 | learned_low_rank_delta | 107 | 0.028354 | 0.013952 | 1 | 0 | 1.01e+06 | 0.032817 | 1.44e+07 |

## Generated figures

- `learning_curves.png`
- `memory_diagnostics.png`
- `router_load.png`

## Independent audits

- Identity/finite/dispatch audit: `reports/phase6/stage1_mechanism_full_taskfix3/identity_audit.json` — 3,000/3,000 rows, no missing/extra/duplicate IDs, no identity mismatches, no failures, no non-finite metrics, and no optimizer-label mismatches.
- Alternating schedule audit: `reports/phase6/stage1_mechanism_full_taskfix3/schedule_audit.json` — all 150 learnable-geometry KAM alternating rows performed at least one geometry update; fixed-geometry KAM rows performed zero; declared 8:1, 32:1, and 128:1 schedules matched their recorded fields.
- The HPG aggregate emitted true Parquet files; the artifact manifest is `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/artifact_manifest.json`.

## Interpretation guardrail

Use this report to locate promising factor/resource combinations and failure modes. Do not promote a configuration from this screen alone; preserve the locked claims, inferential unit, equivalence margins, and held-out data specified by the Phase 6 brief.
