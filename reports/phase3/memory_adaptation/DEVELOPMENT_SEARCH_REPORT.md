# Phase III Development Search Report

Completed rows: **96 / 96**.

The report is descriptive until the preregistered gates are evaluated. Development rows and confirmatory rows are not pooled.

## Metrics

The machine-readable table includes MSE, NMSE, NRMSE, MAE, bias, p90/p95 absolute error, R², correlation, parameter count, wall time, peak VRAM, and optional prequential NLMS endpoints.

## Figures

- `figures/learning_curves.png` — validation learning curves on a log-MSE axis.
- `figures/prediction_true_error.png` — true value, prediction, signed error, and log absolute error.
- `figures/error_distribution_log.png` — log10 absolute-error distributions.
- `figures/primary_effects.png` — paired relative improvements and bootstrap intervals.
- `figures/memory_bank_drift.png` — key/value support-bank movement, with the staged freeze boundary.
- `figures/memory_support_adaptation.png` — support attention and per-support key/value drift over training.
- `figures/memory_train_validation_test.png` — train, validation, and held-out test MSE through the freeze/tuning transition.
- `figures/staged_vs_joint_effects.png` — paired staged-versus-joint validation/test effects.

Causal deletion rows: **90**; support-stability rows: **18**.

## D0 versus DD-b paired effects

| Task | Scale | Endpoint | Pairs | Mean relative improvement | 95% CI | p-value |
|---|---:|---|---:|---:|---:|---:|
| prototype_switch | M | validation_mse | 3 | 0.106 | [-0.152, 0.425] | 0.7561 |
| prototype_switch | M | adaptive_late_post_transition_mse | 1 | -0.357 | [-0.357, -0.357] | 1.0000 |
| prototype_switch | S | validation_mse | 3 | -0.141 | [-0.217, -0.004] | 0.2589 |
| prototype_switch | S | adaptive_late_post_transition_mse | 1 | -0.048 | [-0.048, -0.048] | 1.0000 |
| switching_mackey_glass | M | validation_mse | 3 | -0.289 | [-0.508, -0.053] | 0.2589 |
| switching_mackey_glass | M | adaptive_late_post_transition_mse | 1 | 0.158 | [0.158, 0.158] | 1.0000 |
| switching_mackey_glass | S | validation_mse | 3 | -0.003 | [-0.082, 0.078] | 1.0000 |
| switching_mackey_glass | S | adaptive_late_post_transition_mse | 1 | 0.117 | [0.117, 0.117] | 1.0000 |
| switching_narma | M | validation_mse | 3 | 0.040 | [0.008, 0.098] | 0.2589 |
| switching_narma | M | adaptive_late_post_transition_mse | 1 | 0.042 | [0.042, 0.042] | 1.0000 |
| switching_narma | S | adaptive_late_post_transition_mse | 1 | -0.029 | [-0.029, -0.029] | 1.0000 |
| switching_narma | S | validation_mse | 3 | -0.018 | [-0.099, 0.129] | 1.0000 |

## Staged versus joint DD-b

Positive values favor warmup-then-freeze; negative values favor joint training.

| Task | Scale | Endpoint | Pairs | Mean relative improvement | 95% CI | p-value |
|---|---:|---|---:|---:|---:|---:|
| prototype_switch | M | validation_mse | 4 | 0.146 | [0.082, 0.210] | 0.1279 |
| prototype_switch | M | test_mse | 4 | 0.124 | [-0.109, 0.269] | 0.5027 |
| prototype_switch | S | validation_mse | 4 | -0.703 | [-1.801, 0.231] | 0.7361 |
| prototype_switch | S | test_mse | 4 | -0.751 | [-2.513, 0.299] | 0.8726 |
| switching_mackey_glass | M | validation_mse | 4 | -0.088 | [-0.394, 0.218] | 0.8821 |
| switching_mackey_glass | M | test_mse | 4 | -0.097 | [-0.411, 0.217] | 0.8821 |
| switching_mackey_glass | S | validation_mse | 4 | 0.049 | [-0.054, 0.190] | 0.6177 |
| switching_mackey_glass | S | test_mse | 4 | 0.033 | [-0.109, 0.168] | 1.0000 |
| switching_narma | M | validation_mse | 4 | 0.067 | [-0.052, 0.257] | 0.7456 |
| switching_narma | M | test_mse | 4 | 0.086 | [-0.028, 0.265] | 0.6312 |
| switching_narma | S | validation_mse | 4 | 0.094 | [0.000, 0.245] | 0.2459 |
| switching_narma | S | test_mse | 4 | 0.089 | [-0.012, 0.237] | 0.4818 |
