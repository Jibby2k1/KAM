# Phase 6 overnight implementation and navigation guide

This is the implementation map for the quality-scale four-L4 campaign specified by `KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md`. It is written for both maintainers and LLM reviewers.

## Current campaign

- Submission state: queued on UF HiPerGator on 2026-07-25 at 23:56 EDT.
- HPG checkout: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09`
- Run root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/results/phase6/overnight`
- Report root: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09/reports/phase6/overnight`
- Four-way throttle: exactly one NVIDIA L4 per eligible array row, `%4`.
- Registered work: 60 GPU rows and 45.73 L4 GPU-hours.
- Expected completion: approximately 11:30 AM–1:00 PM EDT on 2026-07-26, plus scheduler delay.
- Immutable graph: `results/phase6/overnight/job_graph.json`

| Node | Slurm ID | Dependency |
|---|---:|---|
| Preflight array | 38052352 | root |
| Preflight gate | 38052353 | after any preflight row state |
| Stage 1 frontier CPU reanalysis | 38052354 | after successful preflight gate |
| Wave 1 array | 38052355 | after Stage 1 frontier |
| Wave 1 aggregate/gate | 38052356 | after any Wave 1 row state |
| Wave 2 manifest controller | 38052357 | after successful Wave 1 gate |
| Wave 2 array | 38052358 | after Wave 2 controller |
| Wave 2 aggregate/gate | 38052359 | after any Wave 2 row state |
| Wave 3 manifest controller | 38052360 | after successful Wave 2 gate |
| Wave 3 array | 38052361 | after Wave 3 controller |
| Final aggregate/report | 38052362 | after any Wave 3 row state |

`afterany` is used only so a gate/report records scientific or infrastructure failures. Every downstream scientific wave depends on successful completion of the preceding gate.

## Read in this order

1. `docs/codex/KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md` — authoritative scientific and execution contract.
2. `configs/phase6/overnight_4xl4_campaign.yaml` — registered factors, budgets, and outcome set.
3. `kam/phase6/overnight_manifest.py` — deterministic 4/32/16/8 row graph and promotion logic.
4. `kam/phase6/overnight_runner.py` — timed language, retrieval, dynamics, adaptation, checkpoint, and causal-diagnostic execution.
5. `kam/phase6/overnight_analysis.py` — gates, Stage 1 Pareto extraction, statistics, Parquet exports, figures, and one-outcome decision.
6. `scripts/phase6_overnight_controller.py` — idempotent CPU controller entry point.
7. `scripts/submit_phase6_overnight_4xl4.sh` and `slurm/phase6_overnight_*.sbatch` — exact HPG dependency graph.
8. `reports/phase6/overnight/README.md` — morning report index and interpretation boundary.

## Row and budget contract

The graph is fixed before submission so all Slurm IDs are known:

- Preflight: 4 rows × 20 minutes = 1.33 GPU-hours.
- Wave 1: 32 rows × 25 minutes = 13.33 GPU-hours.
- Wave 2: 16 rows × 64 minutes = 17.07 GPU-hours.
- Wave 3: 8 rows × 105 minutes = 14.00 GPU-hours.
- Total: 45.73 GPU-hours; ideal four-way occupancy is 11.43 hours before CPU-gate overhead.

Each production row must meet both its minimum token/sample budget and its wall target. Calibration can raise a budget from observed throughput but cannot lower a registered minimum. `PHASE6_OVERNIGHT_SMOKE_SECONDS` is development-only, is recorded in results, and causes a production gate failure.

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

Wave gates reject missing/duplicate/nonfinite/failed rows. They generate the next fixed-size manifest and promote architectures using validation quality plus active compute, wall time, and VRAM. Scientific failures are not retried with new seeds. A failed gate blocks later waves. The final decision is exactly one registered outcome; completion alone never promotes KAM.

## Final metrics and figures

The final report job writes the ten required Parquet tables plus JSONL mirrors. Paired statistics use training seed as the inferential unit and include mean/relative differences, bootstrap 95% CIs, exact paired permutation tests, standardized paired effects, equivalence tests, and Holm correction.

Figures under `reports/phase6/overnight/figures/`:

- `language_learning_curves.png`
- `dynamics_prediction_true_error.png` (true, prediction, and log absolute error)
- `memory_adaptation_freeze.png`
- `resource_quality_pareto.png`
- `adaptation_recovery.png`

Wave 3 also runs reversible memory-branch deletion, KAM top/random/bottom support deletion, key shuffle, value/expert shuffle, and uniform-routing interventions. Results are exported to `deletion_metrics.parquet`.

## Morning commands

One status command:

```bash
ssh hpg 'squeue -j 38052352,38052353,38052354,38052355,38052356,38052357,38052358,38052359,38052360,38052361,38052362 -o "%.18i %.32j %.10T %.10M %R"'
```

One report-rebuild command:

```bash
ssh hpg 'cd /blue/uf-dsi/rvalle1/KAM_repair_2541e09 && source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate && PYTHONPATH=$PWD python scripts/build_phase6_overnight_report.py --run-root results/phase6/overnight --report-root reports/phase6/overnight'
```

Do not interpret queued or running jobs as results. Begin scientific review only when `results/phase6/overnight/final_summary.json` and all seven final reports exist.
