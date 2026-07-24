# LLM handoff: current Phase II state

Goal: decide whether persistent/radial memory improves prediction or adaptation after capacity, compute, paired-stream, and causal-faithfulness controls.

## Verified implementation
- D0/R0/DD/DR/RR plus DD/DR/RR `-v/-a/-b` memory-output modes.
- Continuous switching Mackey–Glass/NARMA with state-preserving schedules and NMSE/NRMSE.
- Exact parameter matching, SQLite-resumable suites, Optuna SQLite/TPE/MedianPruner search, synchronized timing, Phase A interventions, and prequential recovery diagnostics.
- Denser future learning-curve checkpoints, validation prediction exports, and descriptive regression metrics: RMSE, MAE, bias, error quantiles, R², correlation, relative MAE, and tail ratios.

## Verified runs
- 50 stationary paired runs + 50 switching paired runs, five seeds each.
- 80 transition rows across frozen/NLMS/SGD/RLS; all finite.
- 256 full Optuna trials across eight family/task studies; 92 completed and 164 pruned.
- 90 exact-capacity suffix-screen runs at 39,961 parameters; suffix-aware paired statistics are in `results/phase2/dynamic_matrix_pmatched_v2/dynamic_matrix_stats.csv`.
- 1260 five-schedule held-out NLMS transition rows; seed-aggregated statistics are in `results/phase2/heldout_dynamic_matrix_nlms/heldout_stats.csv`.
- 100 five-seed mechanism-language runs: `results/phase2/language_matrix/`.
- Descriptive plots and grouped error metrics: `results/phase2/descriptive_metrics.csv`, `results/phase2/descriptive_metrics_summary.csv`, and `reports/phase2/CHART_MAP.md`.
- Matched timing at 55,576 parameters and capacity smoke at 12,000 parameters.

## Current conclusion
No corrected paired comparison passes the registered CI + Holm gate. Do not claim radial memory works or launch TinyStories/ten-seed confirmation yet.

## Advice requested
Recommend whether to stop the KAM-specific direction or run one narrowly targeted follow-up. If continuing, use DD-b + NLMS versus D0 + NLMS as the primary comparison, with DR-b + NLMS as the radial ablation, and require the same stationary, post-shift, corrected-inference, deletion-faithfulness, support-stability, and overhead gates before any ten-seed confirmation.
