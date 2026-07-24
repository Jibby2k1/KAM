# Phase II descriptive chart map

| Figure | Analytical question | Visual contract | Evidence | Caveat |
|---|---|---|---|---|
| learning_curves_regression.png | Do variants learn at different rates or plateau differently on regression tasks? | Median validation MSE with IQR across completed runs; log y-axis. | metrics.json histories from paired and switching screens. | Legacy runs have sparse checkpoints; future runs now default to 10% evaluation cadence. |
| learning_curves_language.png | Do language/mechanism variants converge differently? | Median validation cross-entropy with IQR. | metrics.json histories from language matrix. | This is a representative mechanism screen, not natural-language modeling. |
| prediction_true_error_switching_mackey_glass.png / prediction_true_error_switching_narma.png | Where do predictions diverge from the true stream and when do errors change sign? | True and prediction above signed error, with regime boundaries. | prequential shift_trace.csv, DR/NLMS/seed 7 fallback selection. | One representative checkpoint; use group metrics for aggregate claims. |
| error_distribution_log.png | Are improvements broad or driven by a few extreme errors? | Boxplots of log10 absolute error by variant and adapter. | prequential shift_trace.csv. | Log scale clips zero at 1e-12; distributions are checkpoint-level observations. |
| post_shift_recovery_curves.png | How quickly do adapters recover after a detected regime transition? | Median/IQR squared loss over 128 post-transition samples. | prequential shift_trace.csv. | Transition alignment is based on observed regime labels for diagnostics only. |
| support_diagnostics.png | Are supports used broadly and aligned with regimes? | Effective support count versus support purity. | shift_metrics.csv. | Purity is descriptive, not causal faithfulness. |

Metric CSVs: `results/phase2/descriptive_metrics.csv` (per checkpoint/seed) and `results/phase2/descriptive_metrics_summary.csv` (aggregated by task, variant, and adapter). Metrics include RMSE, MAE, signed bias, error spread, median/p90/p95/max absolute error, R², correlation, relative MAE, tail ratio, and log10 median absolute error.

QA note: Matplotlib export, dimensions, and nonblank-file checks were run. The local image viewer was unavailable in this sandbox because its filesystem helper failed to initialize loopback networking, so visual inspection was limited to automated file-level QA.
