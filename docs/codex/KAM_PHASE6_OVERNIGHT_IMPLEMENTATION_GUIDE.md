# Phase 6 overnight implementation and navigation guide

This is the implementation map for the quality-scale four-L4 campaign specified by `KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md`. It is written for both maintainers and LLM reviewers.

## Current campaign

- Submission state: the 60-row overnight campaign and its repair graph are complete. Corrected analysis-v2 report job `38201254` completed on UF HiPerGator on 2026-07-28 with exit code 0.
- Corrected analysis code checkout: `/blue/uf-dsi/rvalle1/KAM_analysis_v2_20260728`.
- Immutable evidence root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`
- Corrected analysis view: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight_analysis_v2`
- Corrected report root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight_analysis_v2`
- Four-way throttle: exactly one NVIDIA L4 per eligible array row, `%4`.
- Registered work: 60 GPU rows and 45.73 L4 GPU-hours.
- Final evidence state: 60/60 rows present, zero recorded failures, and all corrected reports and normalized exports generated.
- Immutable graphs: `results/phase6/overnight/{job_graph,timeout_repair_job_graph,calibration_fallback_repair_job_graph}.json`.
- Analysis correction and interpretation boundary: `docs/codex/KAM_PHASE6_ANALYSIS_V2_CORRECTION.md`.
- Historical execution repair details: `docs/codex/KAM_PHASE6_OVERNIGHT_CALIBRATION_REPAIR.md`.

| Historical execution node | Slurm ID | Dependency |
|---|---:|---|
| Preserved evidence | 22 Wave 1 rows | already complete |
| Ten-row Wave 1 repair | 38121449 | root of current graph |
| Exact Wave 1 repair gate | 38121450 | after any repair row state |
| Wave 2 manifest controller | 38121451 | after successful repair gate |
| Wave 2 array | 38121452 | after Wave 2 controller |
| Wave 2 aggregate/gate | 38121453 | after any Wave 2 row state |
| Wave 3 manifest controller | 38121454 | after successful Wave 2 gate |
| Wave 3 array | 38121455 | after Wave 3 controller |
| Final aggregate/report | 38121456 | after any Wave 3 row state |
| Corrected analysis-v2 report | 38201254 | completed against immutable evidence |

`afterany` is used only so a gate/report records scientific or infrastructure failures. Every downstream scientific wave depends on successful completion of the preceding gate.

## Read in this order

1. `docs/codex/KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md` — authoritative scientific and execution contract.
2. `configs/phase6/overnight_4xl4_campaign.yaml` — registered factors, budgets, and outcome set.
3. `kam/phase6/overnight_manifest.py` — deterministic 4/32/16/8 row graph and promotion logic.
4. `kam/phase6/overnight_runner.py` — timed language, retrieval, dynamics, adaptation, checkpoint, and causal-diagnostic execution.
5. `kam/phase6/overnight_analysis.py` — gates, Stage 1 Pareto extraction, statistics, Parquet exports, figures, and one-outcome decision.
6. `scripts/phase6_overnight_controller.py` — idempotent CPU controller entry point.
7. `scripts/submit_phase6_overnight_4xl4.sh` and `slurm/phase6_overnight_*.sbatch` — exact HPG dependency graph.
8. `docs/codex/KAM_PHASE6_ANALYSIS_V2_CORRECTION.md` — corrected result and interpretation boundary.
9. HPG `reports/phase6/overnight_analysis_v2/README.md` — corrected report index.

## Row and budget contract

The graph is fixed before submission so all Slurm IDs are known:

- Preflight: 4 rows × 20 minutes = 1.33 GPU-hours.
- Wave 1: 32 rows × 25 minutes = 13.33 GPU-hours.
- Wave 2: 16 rows × 64 minutes = 17.07 GPU-hours.
- Wave 3: 8 rows × 105 minutes = 14.00 GPU-hours.
- Total: 45.73 GPU-hours; ideal four-way occupancy is 11.43 hours before CPU-gate overhead.

Production language rows use `budget_mode: matched_tokens` and stop at the same registered token budget for every architecture; wall time, throughput, and FLOPs are measured outcomes. Only preflight calibration rows may run to a wall target. This avoids rewarding slower architectures or overtraining faster controls. Cross-architecture calibration fallback remains prohibited. `PHASE6_OVERNIGHT_SMOKE_SECONDS` is development-only, is recorded in results, and causes a production gate failure.

## Persistent-memory lifecycle

The learned-memory behavior is explicit and auditable:

- Fixed-key KAM geometry never receives gradients.
- Joint-SGD learned KAM geometry and algebra update together during the first 80% of registered work.
- ALT KAM uses declared algebra/geometry ratios during the first 80%.
- VP stop-gradient keeps geometry frozen.
- Learned geometry is frozen for the final 20% of training/final tuning.
- Evaluation occurs after this final-tuning freeze.

Every learned-memory subrun records `geometry_steps`, `geometry_freeze_step`, `geometry_frozen_for_final_tuning`, `post_freeze_geometry_drift`, checkpoint-level gate scale, key gradient norm, and value/expert gradient norm. A valid frozen final phase has zero post-freeze drift. The final `memory_adaptation_freeze.png` plots the learning-to-freeze transition.

## Data and identity

Language uses immutable byte tokenization and disjoint 90/5/5 contiguous train/validation/test ranges. The HPG cache does not contain TinyStories, so this run explicitly uses the largest eligible cached legal corpus, `data/tinyshakespeare.txt` (about 1.1 MB), and records the substitution, corpus checksum, tokenizer checksum, and split ranges. A one-sentence fallback is prohibited.

Every row records commit and dirty state, manifest hash and row ID, architecture identity, data/training seeds, precision, GPU/framework versions, total/trainable/active parameters, throughput, estimated compute, wall time, VRAM, checkpoints, and failure category.

## Gates and automatic promotion

Preflight requires all four L4 calibration rows, immutable language checksums and non-overlapping splits, finite/nonconstant dynamics, adequate target variance, and no smoke override. Stage 1 then reanalyzes all 3,000 completed mechanism rows and writes a finite Pareto frontier.

Wave gates reject missing, duplicate, nonfinite, or failed rows. Language promotion is stratified to `small_language` rows only; retrieval and dynamics losses cannot enter the language selector. They generate the next fixed-size manifest using language quality while retaining active compute, wall time, and VRAM as separate system metrics. Scientific failures are not retried with new seeds. A failed gate blocks later waves. The final decision is exactly one registered outcome; completion alone never promotes KAM.

## Final metrics and figures

The final report job writes raw-row and normalized seed-grain Parquet tables plus JSONL mirrors. Language is stratified by wave, lane, task, scale, registered token budget, and exact training seed. Dynamics is additionally stratified by task. Adaptation schedules and tasks are averaged within each base seed. Paired statistics include mean/relative differences, bootstrap 95% CIs, exact paired permutation tests, standardized paired effects, equivalence tests, and within-family Holm correction. The final decision uses only Wave 3 matched-token observations and requires at least six paired seeds, a favorable bootstrap interval, and Holm-adjusted p ≤ 0.05.

The completed legacy adaptation rows declared `rls` for controls and `value_only` for KAM, but the overnight runner actually applied full-model AdamW to every architecture. Analysis v2 records the effective method as `joint_sgd_full_model`, aggregates to five base seeds per architecture, and excludes this unregistered lane from adaptation promotion. Future manifests no longer claim those unimplemented adapters.

Figures under the corrected HPG `reports/phase6/overnight_analysis_v2/figures/`:

- `language_learning_curves_by_wave.png` (wave facets, seed traces, means, and 95% intervals)
- `language_checkpoint_policy_comparison.png` (best, registered-token, and legacy final checkpoints)
- `dynamics_prediction_true_error_comparable.png` (one common task/seed, true/prediction, and log error)
- `memory_adaptation_freeze_learned_variants.png` (learned variants only)
- `resource_quality_wave3_matched.png` (one lane and one registered budget)
- `adaptation_seed_metrics.png` (base-seed points and means)

Wave 3 also runs reversible memory-branch deletion, KAM top/random/bottom support deletion, key shuffle, value/expert shuffle, and uniform-routing interventions. Results are exported to `deletion_metrics.parquet`.

## Verification commands

Confirm the corrected report job:

```bash
ssh hpg 'sacct -j 38201254 --format=JobID,JobName,State,ExitCode,Elapsed'
```

Rebuild analysis v2 without modifying the original evidence:

```bash
ssh hpg 'cd /blue/uf-dsi/rvalle1/KAM_analysis_v2_20260728 && source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate && PYTHONPATH=$PWD python scripts/build_phase6_overnight_report.py --run-root /blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight_analysis_v2 --report-root /blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight_analysis_v2'
```

Use the analysis-v2 summary and reports for scientific review. The original pooled-wave decision is superseded.
