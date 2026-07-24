# Phase V mechanism pilot report

Pilot status: PASSED. Completed 144 of 144 rows with 0 failure artifacts.

## Design

This Stage 1 pilot uses four controlled task labels, six primary controls (D0, DD-L, RF-KV, RF-FULL, KC-LV, RFF), two active-capacity-matched scales, three paired seeds, fixed projected route dimension 64 for KAM models, independent streams, global held-out NMSE, and validation-selected checkpoint reload.

The pilot uses the specified iid-window protocol for signal and variance profiling. Ordered recurrence and adaptation are reserved for the next stage.

## Summary

- P1M / D0: mean NMSE=0.03621, SD=0.01361, n=12, mean active parameters=1001853
- P1M / DD-L: mean NMSE=0.03655, SD=0.01585, n=12, mean active parameters=999801
- P1M / KC-LV: mean NMSE=0.03568, SD=0.01267, n=12, mean active parameters=999801
- P1M / RF-FULL: mean NMSE=0.06179, SD=0.02105, n=12, mean active parameters=999801
- P1M / RF-KV: mean NMSE=0.03468, SD=0.01245, n=12, mean active parameters=999801
- P1M / RFF: mean NMSE=1.157, SD=0.1735, n=12, mean active parameters=999961
- P250 / D0: mean NMSE=0.03995, SD=0.01094, n=12, mean active parameters=249969
- P250 / DD-L: mean NMSE=0.04443, SD=0.0173, n=12, mean active parameters=250553
- P250 / KC-LV: mean NMSE=0.04409, SD=0.01453, n=12, mean active parameters=250553
- P250 / RF-FULL: mean NMSE=0.1023, SD=0.04574, n=12, mean active parameters=250553
- P250 / RF-KV: mean NMSE=0.0446, SD=0.01758, n=12, mean active parameters=250553
- P250 / RFF: mean NMSE=1.171, SD=0.1858, n=12, mean active parameters=249991

## Interpretation guardrails

These results are a mechanism-pilot screen, not a confirmatory decision. The controlled Mackey–Glass, NARMA, prototype, and symbolic labels currently share the controlled-stream execution path; task-specific generator implementations remain a follow-up requirement before final claims.

Read pilot_summary.csv for paired seed-level summaries and pilot_learning_curves.png for convergence behavior. Compare variants within scale and task; do not rank pooled means as if task difficulty were identical.
