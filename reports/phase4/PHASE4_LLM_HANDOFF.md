# ChatGPT handoff: KAM Phase IV

## Ask
Advise on the next experiment stage using the attached repository artifacts. Treat this as a development screen, not confirmatory evidence.

## Current status
- Expected rows: `96`; complete: `96`; failed: `0`.
- Primary report: `reports/phase4/PHASE4_FACTORIAL_REPORT.md`.
- Aggregate table: `results/phase4/factorial_screen/all_metrics.csv`; grouped table: `results/phase4/factorial_screen/summary.csv`.

## Design
- Tasks: prototype switch, switching NARMA, switching Mackey–Glass.
- Conditions: task-specific recurring versus separated controls.
- Variants: D0, jointly trained DD-b, warmup-then-freeze DD-b-staged, and frozen random RF-b.
- Scales: S and M; two paired seeds per cell; validation-selected checkpoints; held-out test metrics.

## Interpretation guardrails
- Relative improvement is `(D0 test MSE - candidate test MSE) / D0 test MSE`.
- This screen does not test all Phase IV hypotheses and cannot establish stochasticity, coordinate mismatch, or causal support use.
- Inspect `figures/learning_curves.png`, `figures/prediction_true_error.png`, and `figures/memory_drift.png` alongside the grouped table.

## Questions for advice
1. Which observed task × condition × variant effects merit Stage C freeze-policy search?
2. Should the next screen prioritize noise/observability controls or full-path/drift-triggered freezing?
3. What minimum paired seed/stream design would make the top mechanism claim credible?
4. Which negative result would justify simplifying to generic adaptive readout?

