# Phase 6 Stage 1 profile — superseded geometry-coverage run

## Status

This execution is **superseded for scientific interpretation**.

- HPG array: `38036789`
- HPG aggregate: `38036793`
- Structural execution result: 64/64 rows passed
- Scientific coverage result: **invalid**

The first profile manifest used a generic plural-to-singular field conversion that mapped `geometries` to `geometrie`. Consequently, every row carried `geometry=None`, and the runner fell back to its default learned geometry. The run therefore did not test the declared fixed/data/k-means/farthest/learned geometry grid.

The row outputs are retained at `results/phase6/stage1_mechanism/hpg_runs_profile_geometry_gap/` for audit only. They must not be used to promote configurations or justify a scientific conclusion.

## Remediation

The manifest builder now maps `geometries` explicitly to `geometry` and asserts coverage of all six declared modes. The corrected rerun is separate:

- HPG array: `38037900`
- HPG aggregate: `38037901`
- Corrected manifest: `results/phase6/stage1_mechanism/manifests/profile.jsonl`
- Corrected HPG result root: `results/phase6/stage1_mechanism/hpg_runs_profile_geometry_fix/`

