# Phase 6 overnight queued handoff

- Status: timeout repair queued; scientific reports pending.
- Validation: 65/65 local tests, 63/63 initial HPG-environment tests, and 3/3 focused HPG repair tests passed.
- HPG test duration: 663.70 seconds.
- Initial Wave 1: 20/32 rows passed; 12 rows reached the three-hour infrastructure limit; Wave 2/3 did not start.
- Repair root/final Slurm jobs: `38087856` / `38087863`.
- Allocation: 60 rows, one NVIDIA L4 per row, `%4`, 45.73 registered GPU-hours.
- HPG run root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`
- HPG report root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight`
- Repair preserves 20 completed rows and reruns only the 12 missing rows with corrected memory/retrieval budgets.

The graph is dependency-gated. Any failed calibration or scientific wave prevents later scientific work; aggregate jobs still run after any array state so failures can be documented.

No result or promotion claim should be made until `final_summary.json` and the seven final reports exist.
