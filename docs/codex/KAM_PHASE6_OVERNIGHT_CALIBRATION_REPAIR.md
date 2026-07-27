# Phase 6 overnight architecture-calibration repair

## Current status

The second selective repair was submitted to UF HiPerGator on 2026-07-27 at 8:56 AM EDT. It preserves 22 complete Wave 1 rows and reruns only the 10 rows still missing. Four repair rows started immediately on separate NVIDIA L4 GPUs; six are held by the intentional `%4` throttle. Wave 2, Wave 3, and final reporting are dependency-gated behind exact row-identity checks.

This is execution status, not a scientific result. Do not advise model promotion until `results/phase6/overnight/final_summary.json` and all seven final reports exist.

## Root cause

The first repair correctly reduced memory sizes and retrieval example floors, but `_resolve_budget` still allowed rows without an architecture-specific calibration rate to inherit a generic rate measured on another architecture/lane. On the recorded calibration this caused:

- KAM language rows registered for 50M tokens to resolve to about 111M tokens.
- Retrieval rows registered for about 5M processed-token equivalents to resolve to about 535K examples.

The six-hour jobs therefore exhausted their Slurm limit while performing unregistered excess work. Two T-MEMTOK language rows completed; seven later rows timed out and three active retrieval rows were stopped when the superseded graph was canceled. No scientific row failure was observed.

## Correction

- Calibration is now accepted only from the same architecture and compatible lane.
- `language_replication` may reuse the same architecture's `language` rate.
- An uncalibrated architecture/lane runs its explicit registered minimum plus target duration; generic cross-architecture token/sample rates are ignored.
- Missing rows receive `repair_revision: 2`, a new content-addressed row ID, and both immediate and original provenance links.
- Only output JSON with `status: pass` counts as completed evidence when constructing a repair.
- Wave 1 repair has an eight-hour limit. Wave 2/3 have 24-hour limits to safely accommodate their larger registered floors; these are ceilings, not requested work increases.

Audited Wave 1 revision-2 targets are exactly 50M tokens for each of the six language rows. Retrieval targets are 19,532, 9,766, 4,883, and 39,063 examples according to registered sequence-length-equivalent floors.

## Immutable records

- Full 32-row Wave 1 manifest SHA-256: `f1e7fd10a805132474ebfed7ee8f4b9b146949bf53c66f45dfb6931245447299`
- Ten-row repair manifest SHA-256: `aad17c74bf604df0c4e750ad36b9b00e98d2be6d3ce2c15a00dd361a0391a677`
- Tested/deployed commit: `e777ddcfe90171e03c7511e4ca7e49c1b4a13e51`
- Clean code checkout: `/blue/uf-dsi/rvalle1/KAM_calibration_repair_e777ddc`
- Preserved run root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`

Files:

- `results/phase6/overnight/manifests/wave1_pre_calibration_fallback_repair.jsonl`
- `results/phase6/overnight/manifests/wave1.jsonl`
- `results/phase6/overnight/manifests/wave1_calibration_fallback_repair.jsonl`
- `results/phase6/overnight/calibration_fallback_repair_job_graph.json`

## Replacement graph

| Node | Slurm ID |
|---|---:|
| Ten-row Wave 1 repair | 38121449 |
| Exact Wave 1 gate | 38121450 |
| Wave 2 controller | 38121451 |
| Wave 2 array | 38121452 |
| Wave 2 aggregate/gate | 38121453 |
| Wave 3 controller | 38121454 |
| Wave 3 array | 38121455 |
| Final aggregate/report | 38121456 |

The superseded graph `38087856–38087863` was canceled exactly. Its two completed revision-1 rows remain valid and are preserved; partial checkpoints are not combined with revision-2 runs.

## Validation and status

- Complete local suite: 67 passed.
- Focused clean-checkout HPG suite: 5 passed.
- Shell syntax, Python compilation, dry-run submission, exact commit identity, manifest uniqueness, and resolved budgets passed.

```bash
ssh hpg 'squeue -j 38121449,38121450,38121451,38121452,38121453,38121454,38121455,38121456 -o "%.18i %.32j %.10T %.10M %.10l %R"'
```
