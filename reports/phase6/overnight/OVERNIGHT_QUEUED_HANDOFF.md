# Phase 6 overnight queued handoff

- Status: architecture-calibration repair queued; scientific reports pending.
- Validation: 67/67 local tests and 5/5 focused clean-checkout HPG repair tests passed.
- HPG test duration: 663.70 seconds.
- Current Wave 1: 22/32 rows passed; 10 revision-2 rows are queued/running; Wave 2/3 have not started.
- Repair root/final Slurm jobs: `38121449` / `38121456`.
- Allocation: 60 rows, one NVIDIA L4 per row, `%4`, 45.73 registered GPU-hours.
- HPG run root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`
- HPG report root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight`
- Repair preserves 22 completed rows and reruns only the 10 missing rows with architecture-specific calibration or registered minimum budgets.

The graph is dependency-gated. Any failed calibration or scientific wave prevents later scientific work; aggregate jobs still run after any array state so failures can be documented.

No result or promotion claim should be made until `final_summary.json` and the seven final reports exist.
