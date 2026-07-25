# Phase V Stage 2 repair and rerun handoff

## Status

The repository-level validity defects are repaired. Scientific results are
still blocked until the affected HiPerGator rows are rerun and every
machine-readable gate passes.

This repair:

- preserves every valid requested-seed NARMA stream exactly;
- deterministically retries only a candidate that fails the unchanged stream
  quality gate;
- records `requested_seed`, `realized_seed`, `seed_attempt`, the retry limit,
  quality metrics, and gate outcomes for every controlled NARMA split;
- retains the same audit trail inside held-out stream metrics;
- resolves Stage 2D architectures with the actual per-cell vocabulary: 13
  tokens for S0/S1 and 16 for S2;
- keeps exact manifest-to-runtime capacity equality and the cross-variant
  capacity-spread limit at 1%;
- restores the executable bit on the Stage 2 HiPerGator submission script.

Do not change the NARMA equation, loosen a quality threshold, reuse stale
Stage 2D checkpoints, or draw scientific conclusions before the sequence below
is complete.

## Binding next actions

### 1. Synchronize and verify

On HiPerGator, pull remote `main` into a clean worktree and record its commit:

```bash
git switch main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
pytest -q
python -m kam.phase5.stage2_manifest \
  --output-dir results/phase5/stage2/manifests
git diff --exit-code -- results/phase5/stage2/manifests
```

Stop if the worktree is unexpectedly dirty, tests fail, or manifest
regeneration differs from the committed manifests.

### 2. Reuse only the valid prior artifacts

The failed Stage 2B rows must train from scratch:

```text
321,361,401,441
```

They are the four paired variants for NARMA cell `P250000`, seed index 1,
requested training seed `41271`.

The affected Stage 2C rows already have valid training checkpoints. Rerun them
with `--resume` so only missing held-out evaluation is completed:

```text
822,826,830,834,838,883,887,891,895,899
```

These are the five variants in `F5`, seed index 2, and the five variants in
`F8`, seed index 3.

Submit these exact indices through `slurm/phase5_stage2_array.sbatch`, with
`STAGE2_MANIFEST` and `STAGE2_RUN_ROOT` pointing to the original reassessment
manifest and sub-study run root. Keep `--resume`; Stage 2B failure directories
are incomplete and will retrain, while Stage 2C complete checkpoints will be
reused for held-out evaluation.

Before submission, print and verify the selected manifest rows:

```bash
python - <<'PY'
from pathlib import Path
from kam.phase4.table import read_table

root = Path("results/phase5/stage2/manifests")
selected = {
    "stage2B_capacity": {321, 361, 401, 441},
    "stage2C_factorial": {822, 826, 830, 834, 838, 883, 887, 891, 895, 899},
}
for stage, indices in selected.items():
    rows = read_table(root / f"{stage}.jsonl")
    for index in sorted(indices):
        row = rows[index]
        assert row["row_id"] == index
        print(stage, index, row["run_id"], row["seed"])
PY
```

### 3. Rerun all Stage 2D rows in a new root

Run indices `0-59` from the regenerated
`stage2D_symbolic.jsonl`. Use a new, empty run root labeled with the repair
commit. Do not use `--resume` against the old Stage 2D run root: its
checkpoints were constructed with the wrong vocabulary-dependent architecture.

The new manifest contract is:

| Cell | Vocabulary | Shared active target | D0 count | Other paired counts |
|---|---:|---:|---:|---:|
| S0 | 13 | 1,000,493 | 1,000,297 | 1,000,493 |
| S1 | 13 | 1,000,493 | 1,000,297 | 1,000,493 |
| S2 | 16 | 1,001,552 | 1,001,380 | 1,001,552 |

The worst paired spread is approximately 0.0196%, safely below 1%. Exact
runtime equality with each row's `resolved_active_parameters` is still
required.

### 4. Audit artifacts before aggregation

For Stage 2B, Stage 2C, and the new Stage 2D root, verify:

- expected `metrics.json` count;
- expected `heldout_metrics.json` count;
- zero `failure.json` files;
- every `phase5_pilot_checks` value is true;
- every runtime active count equals `resolved_active_parameters`;
- every paired capacity error is at most 0.01;
- every controlled split passes the stream-quality checks.

For the 14 repaired NARMA rows, additionally extract the seed audit trail from
`metrics.json`:

- training/validation/test/prequential:
  `data_metadata.stream_metadata`;
- held-out generated splits:
  `heldout_stream_metrics[*].stream_generation_metadata`.

Confirm that variants paired within the same scientific cell use identical
requested and realized seeds. A nonzero `seed_attempt` is allowed only when the
requested candidate failed the predeclared gate.

The known deterministic replacements should be:

| Stage/cell | Split | Requested seed | Realized seed | Attempt |
|---|---|---:|---:|---:|
| Stage 2B / P250000 / seed index 1 | campaign train | 41,271 | 4,257,143,044 | 1 |
| Stage 2C / F5 / seed index 2 | held-out stream 1 validation | 15,079,651 | 177,899,455 | 1 |
| Stage 2C / F8 / seed index 3 | held-out stream 0 validation | 8,080,344 | 4,084,619,730 | 1 |

Treat any additional retry as an audit item: preserve it, verify that attempt
zero genuinely failed the gate, and explain it in the reassessment report.

### 5. Aggregate only after row-level validation

Run `kam.phase5.stage2_aggregate` separately for the repaired Stage 2B,
Stage 2C, and new Stage 2D roots with expected counts 480, 1080, and 60.
Require `stage2_checks.json` to report `"passed": true` for all three.

Do not rerun Stage 2A; its 450 rows already passed and this repair does not
change its realized streams unless a future audit shows otherwise.

### 6. Update evidence and decide

After every gate passes:

1. replace the blocked Stage 2B, Stage 2C, and Stage 2D reports with aggregates
   from the repaired roots;
2. update `PHASE5_STAGE2_REASSESSMENT_BLOCKERS.md` with commit, Slurm job IDs,
   run roots, completion counts, retry audit, and final gate results;
3. rerun paired effects, bootstrap intervals, permutation tests, and Holm
   adjustment;
4. compare conclusions only within task and capacity/factor cells;
5. create a new LLM decision memo that explicitly distinguishes
   implementation validity from scientific evidence;
6. authorize Stage 3 only if the repaired statistics satisfy the
   preregistered promotion criteria.

If any gate fails, stop at that stage, preserve the failure artifact, diagnose
the specific row, and do not aggregate around it.
