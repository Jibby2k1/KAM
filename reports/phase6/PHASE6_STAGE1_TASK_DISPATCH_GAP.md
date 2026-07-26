# Phase 6 Stage 1 — superseded task-dispatch run

The geometry-corrected profile (`38038386`, aggregate `38038387`) completed 64/64 rows and produced valid artifacts, but it is superseded for scientific interpretation after a runner audit found two dispatch defects:

- Stage 1 rows labeled `mqar` used the NARMA fallback instead of the MQAR retrieval fixture.
- Stage 1 optimizer labels were recorded but the runner used the same joint Adam loop for all labels.

The run is retained at `results/phase6/stage1_mechanism/hpg_runs_profile_final/` for execution/resource auditing only. The runner now dispatches MQAR explicitly, executes the declared optimizer modes, and uses task-specific Stage 2/4 fixtures. The first replacement (`38039123` / `38039124`) failed uniformly because the HPG checkout lacked the updated `kam.data.phase6` export; no scientific rows were produced. The next replacement (`38039556` / `38039557`) exposed six T-WIDE rows that fitted the ridge solve on post-readout features; that shape bug was fixed and regression-tested. The passing task-aware profile (`38040026` / `38040027`) matched the manifest with zero failures and zero dispatch mismatches, but its short fidelity never reached geometry phases for alternating schedules; use it for task/optimizer identity only, not geometry-learning interpretation.

The first full deployment (`38040418` / `38040419`) was intentionally canceled after 1,315 partial rows when its completed alternating rows were audited: the short `1–4` step budgets produced zero geometry updates. Its partial outputs remain under `results/phase6/stage1_mechanism/hpg_runs_full_taskfix2/` for audit only. The runner now guarantees at least one geometry update when geometry parameters exist and records declared versus effective schedule steps. The corrected full campaign is array `38042710` with aggregate `38042711`, under HPG root `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/`.
