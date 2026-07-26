# Phase 6 overnight queued handoff

- Status: queued; scientific reports pending.
- Validation: 63/63 local tests and 63/63 HPG-environment tests passed.
- HPG test duration: 663.70 seconds.
- Root/final Slurm jobs: `38052352` / `38052362`.
- Allocation: 60 rows, one NVIDIA L4 per row, `%4`, 45.73 registered GPU-hours.
- HPG run root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`
- HPG report root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight`
- Expected completion: 11:30 AM–1:00 PM EDT on 2026-07-26, plus scheduler delay.

The graph is dependency-gated. Any failed calibration or scientific wave prevents later scientific work; aggregate jobs still run after any array state so failures can be documented.

No result or promotion claim should be made until `final_summary.json` and the seven final reports exist.
