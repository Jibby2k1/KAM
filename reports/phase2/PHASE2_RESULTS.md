# Phase II Results

## Technical summary

The workspace now contains a verified Phase II foundation plus **100 baseline paired training runs**: 50 stationary dynamical runs and 50 continuous switching runs, across five variants and five seeds. A separate exact-capacity suffix screen adds **90 runs** across nine variants, two switching systems, and five seeds, all at 39,961 parameters. The evaluator produced **80 all-adapter transition rows** plus **1260 held-out NLMS transition rows**; the full Optuna search produced **256 trials** across eight SQLite studies (92 complete). The reporting layer now adds **7 descriptive figures** and a grouped error-metric table.

No corrected paired comparison currently passes the confidence-interval and Holm gate (0 passes). The evidence therefore does not promote a radial-memory architecture or authorize the conditional TinyStories stage.

## Paired stationary and switching evidence

Positive mean improvement means the candidate’s MSE was lower than the named baseline. Confidence intervals are paired bootstrap intervals over five training seeds; p-values are Holm-adjusted across the declared comparisons.

| Task | Baseline | Candidate | Pairs | Mean improvement | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| mackey_glass | D0 | R0 | 5 | 0.000198515 | [-4.12655e-05, 0.000381465] | 1 |
| mackey_glass | DD | RR | 5 | 4.31711e-05 | [-0.000102875, 0.000168107] | 1 |
| mackey_glass | DD | DR | 5 | 2.35767e-05 | [-3.75007e-06, 4.94395e-05] | 1 |
| narma | D0 | R0 | 5 | -0.0166418 | [-0.0503502, 0.0337516] | 1 |
| narma | DD | RR | 5 | -0.0264436 | [-0.0650398, 0.0129321] | 1 |
| narma | DD | DR | 5 | 0.00286424 | [-0.00353981, 0.0083133] | 1 |
| switching_mackey_glass | D0 | R0 | 5 | 0.000157989 | [-8.56606e-05, 0.000426315] | 1 |
| switching_mackey_glass | DD | RR | 5 | 0.000113721 | [-0.000223899, 0.00046816] | 1 |
| switching_mackey_glass | DD | DR | 5 | 0.000188357 | [0.000108276, 0.000268359] | 0.391161 |
| switching_narma | D0 | R0 | 5 | -0.0297902 | [-0.0701932, 0.0106127] | 1 |
| switching_narma | DD | RR | 5 | -0.0342641 | [-0.103038, 0.0382878] | 1 |
| switching_narma | DD | DR | 5 | 0.000212033 | [-0.000801179, 0.00144442] | 1 |

The only unadjusted-looking positive signal in the switching table is not sufficient after Holm correction. The report treats that as a shortlist signal, not a claim.

## Exact parameter-matched dynamic screen

The suffix-aware screen tests persistent values, routes, and both pathways, plus radial memory at the same output mode. Every row below uses five paired seeds and exactly 39,961 trainable parameters; no corrected comparison passes.

| Task | Claim | Baseline | Candidate | Pairs | Parameters | Mean improvement | 95% CI | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| switching_mackey_glass | radial_context | D0 | R0 | 5 | 39961 | 0.000314873 | [5.42174e-05, 0.000575529] | 1 |
| switching_mackey_glass | persistent_values | D0 | DD-v | 5 | 39961 | -0.000187347 | [-0.000995105, 0.000740309] | 1 |
| switching_mackey_glass | persistent_routes | D0 | DD-a | 5 | 39961 | -8.64035e-05 | [-0.00083274, 0.000654748] | 1 |
| switching_mackey_glass | persistent_both | D0 | DD-b | 5 | 39961 | -8.55512e-06 | [-0.000775232, 0.00077032] | 1 |
| switching_mackey_glass | radial_memory_values | DD-v | DR-v | 5 | 39961 | 3.97054e-05 | [-0.000173937, 0.000253348] | 1 |
| switching_mackey_glass | radial_memory_routes | DD-a | DR-a | 5 | 39961 | 4.56521e-05 | [-6.69418e-05, 0.000156262] | 1 |
| switching_mackey_glass | radial_memory_both | DD-b | DR-b | 5 | 39961 | 0.000154008 | [-2.36448e-05, 0.000360045] | 1 |
| switching_mackey_glass | radial_context_with_memory | DD-b | RR-b | 5 | 39961 | 0.000125905 | [-0.000215832, 0.00048978] | 1 |
| switching_narma | radial_context | D0 | R0 | 5 | 39961 | -0.0320037 | [-0.0646592, 0.00065179] | 1 |
| switching_narma | persistent_values | D0 | DD-v | 5 | 39961 | -0.000210709 | [-0.0408017, 0.0578663] | 1 |
| switching_narma | persistent_routes | D0 | DD-a | 5 | 39961 | 0.0112978 | [-0.0351083, 0.0656902] | 1 |
| switching_narma | persistent_both | D0 | DD-b | 5 | 39961 | 0.0114535 | [-0.0328743, 0.0669669] | 1 |
| switching_narma | radial_memory_values | DD-v | DR-v | 5 | 39961 | 0.00460571 | [-0.00362515, 0.0142441] | 1 |
| switching_narma | radial_memory_routes | DD-a | DR-a | 5 | 39961 | 0.00155023 | [-0.00221218, 0.0070387] | 1 |
| switching_narma | radial_memory_both | DD-b | DR-b | 5 | 39961 | -0.00404143 | [-0.0127413, 0.00116278] | 1 |
| switching_narma | radial_context_with_memory | DD-b | RR-b | 5 | 39961 | -0.0385011 | [-0.107697, 0.0360787] | 1 |

## Formal-language generalization

The variable-copy checkpoint uses sinusoidal positions and trains on payloads sampled from 8–64. At unseen lengths, copied-token accuracy and exact-sequence accuracy remain low; the task is not yet a successful length-generalization result.

| Payload length | Copied-token accuracy | Exact sequence accuracy | Cross-entropy |
|---:|---:|---:|---:|
| 8.0 | 0.508789 | 0.0078125 | 1.36623 |
| 64.0 | 0.224121 | 0 | 2.29113 |
| 80.0 | 0.0826172 | 0 | 3.55106 |
| 96.0 | 0.0726725 | 0 | 3.70095 |
| 128.0 | 0.072876 | 0 | 3.60563 |
| 192.0 | 0.0657145 | 0 | 3.68773 |

The bounded Dyck-2 checkpoint was trained to depth 8. Grammar-validity remains zero at depths 8–16 in this bounded run, so it is a diagnostic failure rather than evidence of hierarchical generalization.

| Depth | Token accuracy | Grammar-valid fraction |
|---:|---:|---:|
| 8.0 | 0.471191 | 0 |
| 10.0 | 0.447656 | 0 |
| 12.0 | 0.431966 | 0 |
| 16.0 | 0.421387 | 0 |

## Five-seed mechanism-language screen

The representative mechanism screen completed 100 runs across MQAR, variable copy, Dyck-2, and reusable-regime grammar. These are mechanism diagnostics, not evidence to override the failed dynamic-memory gate.

| Task | Variant | Seeds | Cross-entropy | Accuracy |
|---|---:|---:|---:|---:|
| dyck2 | D0 | 5 | 0.722466 | 0.661426 |
| dyck2 | DD | 5 | 0.720976 | 0.662891 |
| dyck2 | DR | 5 | 0.720339 | 0.662988 |
| dyck2 | R0 | 5 | 0.721349 | 0.661133 |
| dyck2 | RR | 5 | 0.715363 | 0.666797 |
| mqar | D0 | 5 | 2.11733 | 0.126172 |
| mqar | DD | 5 | 2.0964 | 0.132031 |
| mqar | DR | 5 | 2.09716 | 0.13125 |
| mqar | R0 | 5 | 2.11642 | 0.127344 |
| mqar | RR | 5 | 2.09639 | 0.130469 |
| regime | D0 | 5 | 2.66169 | 0.162153 |
| regime | DD | 5 | 2.65131 | 0.162277 |
| regime | DR | 5 | 2.65003 | 0.16307 |
| regime | R0 | 5 | 2.68427 | 0.149405 |
| regime | RR | 5 | 2.66934 | 0.154291 |
| variable_copy | D0 | 5 | 2.81529 | 0.0853497 |
| variable_copy | DD | 5 | 2.81363 | 0.0877221 |
| variable_copy | DR | 5 | 2.81352 | 0.0874864 |
| variable_copy | R0 | 5 | 2.828 | 0.074662 |
| variable_copy | RR | 5 | 2.8262 | 0.0792499 |

## Prequential adaptation and support diagnostics

Each transition follows predict → score → reveal → update. Recovery steps are the first rolling window within 10% of the late segment loss. Support purity is descriptive alignment between argmax support assignments and hidden regime labels; labels are never fed to the model.

| Task | Adapter | Transitions | Early loss | Late loss | Recovery steps | Support purity | Effective supports |
|---|---:|---:|---:|---:|---:|---:|---:|
| switching_mackey_glass | frozen | 10 | 0.0105786 | 0.00906878 | 35.5 | 0.666835 | 31.6768 |
| switching_mackey_glass | nlms | 10 | 0.00366044 | 0.00198526 | 98.7 | 0.666835 | 31.6768 |
| switching_mackey_glass | rls | 10 | 0.00154623 | 0.000743095 | 77 | 0.666835 | 31.6768 |
| switching_mackey_glass | sgd | 10 | 0.00271515 | 0.0014342 | 58.2 | 0.666835 | 31.6768 |
| switching_narma | frozen | 10 | 0.342251 | 0.394878 | 0.4 | 0.664735 | 31.7934 |
| switching_narma | nlms | 10 | 0.329091 | 0.36528 | 3.8 | 0.664735 | 31.7934 |
| switching_narma | rls | 10 | 0.264971 | 0.290387 | 2.1 | 0.664735 | 31.7934 |
| switching_narma | sgd | 10 | 0.340839 | 0.374783 | 5.8 | 0.664735 | 31.7934 |

All adaptation transition rows were finite: **True**. Support diagnostics and deletion curves are saved separately because they are checkpoint-level diagnostics, not independent training replicates.

## Descriptive prediction and error evidence

The new plots separate optimization behavior, pointwise prediction behavior, tail risk, and post-shift recovery. Learning curves show medians with interquartile bands; legacy runs contain only three recorded checkpoints, while future runs now default to a denser 10%-of-budget evaluation cadence.

![Regression learning curves](figures/learning_curves_regression.png)

The language/mechanism panels use the same median/IQR convention with cross-entropy rather than MSE, keeping optimization curves comparable without conflating the task units.

![Language and mechanism learning curves](figures/learning_curves_language.png)

The representative true-versus-prediction panels make signed error and regime-boundary behavior visible for one DR/NLMS/seed-7 stream; aggregate claims should use the descriptive metric CSV and paired tables.

![Mackey–Glass true versus prediction and signed error](figures/prediction_true_error_switching_mackey_glass.png)

![NARMA true versus prediction and signed error](figures/prediction_true_error_switching_narma.png)

The log absolute-error distributions expose whether a method’s average improvement is broad or dominated by tail behavior; the metric export adds p90/p95/max error, bias, R², correlation, relative MAE, and p95-to-median tail ratio.

![Log absolute-error distributions](figures/error_distribution_log.png)

The recovery curves align squared loss at detected regime transitions, while the support plot separates utilization from descriptive regime alignment; neither should be read as causal attribution by itself.

![Post-shift recovery curves](figures/post_shift_recovery_curves.png)

![Support utilization and regime alignment](figures/support_diagnostics.png)

## Held-out schedule coverage

The exact-capacity matrix was evaluated on five independent schedule/stream-seed combinations per checkpoint with the primary NLMS adapter, yielding 1260 transition rows. Schedule-level results were averaged within training seed before paired inference; no corrected comparison passes.

| Task | Claim | Baseline | Candidate | Seed pairs | Mean late-loss improvement | 95% CI | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| switching_mackey_glass | persistent_memory | D0 | DD-b | 5 | 0.00175335 | [0.000754794, 0.00266866] | 0.391161 |
| switching_mackey_glass | radial_memory | DD-b | DR-b | 5 | -0.000454553 | [-0.000990133, -3.69649e-05] | 0.632937 |
| switching_mackey_glass | radial_context | D0 | R0 | 5 | -0.000194041 | [-0.00187432, 0.00191289] | 1 |
| switching_narma | persistent_memory | D0 | DD-b | 5 | 0.0117368 | [-0.01385, 0.0347713] | 1 |
| switching_narma | radial_memory | DD-b | DR-b | 5 | -0.00232136 | [-0.00817339, 0.00227162] | 1 |
| switching_narma | radial_context | D0 | R0 | 5 | -0.00796986 | [-0.021898, 0.00391865] | 1 |

## Capacity, timing, and search controls

The capacity smoke matched D0 and DR exactly at 12,000 parameters. The timing benchmark matched all five language variants at 55,576 parameters and measured synchronized AMP forward latency with median/IQR/P90 and peak VRAM.

| Sequence length | Variant | Parameters | Median ms | IQR ms | P90 ms | Peak MB |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | D0 | 55576 | 0.465329 | 0.0422236 | 0.545737 | 10.0693 |
| 32 | DD | 55576 | 0.640381 | 0.015786 | 0.699355 | 11.1528 |
| 32 | DR | 55576 | 0.83364 | 0.0196248 | 0.86054 | 11.9849 |
| 32 | R0 | 55576 | 0.659401 | 0.0406551 | 0.778743 | 10.9951 |
| 32 | RR | 55576 | 1.02447 | 0.0184356 | 1.03309 | 12.9102 |
| 64 | D0 | 55576 | 0.443713 | 0.00868429 | 0.450021 | 12.4639 |
| 64 | DD | 55576 | 0.62869 | 0.0182366 | 0.645721 | 14.4888 |
| 64 | DR | 55576 | 0.858287 | 0.102474 | 1.34317 | 15.8364 |
| 64 | R0 | 55576 | 0.650412 | 0.0136499 | 0.676705 | 14.8115 |
| 64 | RR | 55576 | 1.00999 | 0.0168565 | 1.02614 | 18.1836 |

The full Optuna search used SQLite storage, TPE sampling, and MedianPruner configuration; it completed 92/256 trials, with pruned trials recorded in the study database.

## Required gates and decision

1. Stationary degradation ≤5%: not established for a promoted architecture after matched controls.
2. Post-shift improvement ≥15%: not established by the five-schedule held-out NLMS screen after paired inference.
3. Corrected CI excludes zero: **not passed**.
4. Top-support deletion beats random: deletion curves exist, but cross-seed causal faithfulness is not yet passed.
5. Support noncollapse: short-run utilization is healthy in memory models, but cross-seed stability is not passed.
6. Overhead: measured and reported; radial/memory variants carry measurable latency and VRAM cost.

Decision: **continue to development-stage switching and capacity-matched diagnostics; do not promote to ten-seed confirmation or TinyStories.**

## Artifacts and open work

- Paired stationary: `results/phase2/paired_screen/`
- Paired switching: `results/phase2/switching_paired_screen/`
- Exact parameter-matched suffix screen: `results/phase2/dynamic_matrix_pmatched_v2/`
- Five-schedule held-out NLMS screen: `results/phase2/heldout_dynamic_matrix_nlms/`
- Prequential recovery: `results/phase2/switching_adaptation/`
- Phase A switching reanalysis: `results/phase2/switching_reanalysis.csv` and its deletion/raw files
- Full Optuna SQLite search: `results/phase2/search_full/optuna_search/`
- Matched timing: `results/phase2/timing/`
- Variable-copy generalization: `results/phase2/variable_copy_generalization_v3/generalization.csv`
- Dyck-2 generalization: `results/phase2/dyck_generalization/generalization.csv`
- Figures: `reports/phase2/figures/`
- Descriptive metrics: `results/phase2/descriptive_metrics.csv`
- Descriptive metric summary: `results/phase2/descriptive_metrics_summary.csv`
- Chart map and QA notes: `reports/phase2/CHART_MAP.md`

Remaining gated work: ten-seed confirmation only after the registered gates pass, and conditional TinyStories only after a passing dynamic-memory gate.
