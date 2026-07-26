# Phase 6 results layout

Each stage uses immutable JSONL manifests and independently rerunnable row outputs. Stage 0 local and HPG evidence is under `stage0/`. Profile/full stage directories are created by `kam.phase6.manifest` and `kam.phase6.run_array`; aggregation writes `all_metrics.jsonl`, a machine-readable summary, and a stage report without shared SQLite state.

Parquet export is optional and should be enabled only when the HPG environment provides a validated Parquet engine; the JSONL artifacts remain the canonical interchange format.
