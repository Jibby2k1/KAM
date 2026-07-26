# Phase 6 overnight timeout repair handoff

## Status

The initial campaign passed preflight and the 3,000-row Stage 1 frontier reanalysis. Wave 1 produced 20 valid rows, but 12 rows reached the original three-hour Slurm limit before satisfying their registered work floors. Gate `38052356` correctly failed with 20/32 outputs, so Wave 2 and Wave 3 did not run.

The repair was submitted on 2026-07-26 at 5:31 PM EDT. It preserves the 20 completed rows and reruns only the 12 missing rows.

## Root cause and correction

Two manifest quantities were not scaled to the actual work represented:

1. T-MEMTOK used 1,024 persistent tokens, while the registered practical control should use 32.
2. Retrieval used a flat 200,000-example floor. At sequence length 1,024 this implied about 205 million processed tokens, far beyond a 25-minute screen.

The amendment makes the following prospective and repair changes:

- T-MEMTOK: 32 memory tokens.
- Learned/ALT/VP KAM language: 1,024 supports.
- Long-sequence KAM retrieval: 1,024 supports.
- Retrieval floors: at least five million processed tokens, converted to examples by sequence length, with a minimum of 4,096 examples.
- Explicit Wave 1 repair provenance: `repair_revision`, `repair_reason`, and `supersedes_row_id`.
- Exact gate identity: every wave now requires equality between manifest row IDs and output row IDs, not only matching counts.
- Wall limits: six hours for Wave 1 repair and eight hours for Wave 2/3. Scientific minimum token budgets remain unchanged for language.

## Immutable manifests

- Superseded Wave 1 SHA-256: `07f74abd71b5e5cfc793b4c50adb66e905eb2a4a9953e63194dc9ffb5de9630c`
- Amended full Wave 1 SHA-256: `a6cf8391fe22a27de32c7f69acf956a62e4ea309ac0b67714bb32048efce4716`
- Twelve-row repair SHA-256: `50b3041d5156e3ceaa3e44d5e53656f2666c2a82d0167008e3c116cfe5529d7a`

Files:

- `results/phase6/overnight/manifests/wave1_pre_timeout_repair.jsonl`
- `results/phase6/overnight/manifests/wave1.jsonl`
- `results/phase6/overnight/manifests/wave1_timeout_repair.jsonl`
- `results/phase6/overnight/wave1_gate_initial_failure.json`

## Replacement Slurm graph

| Node | Job ID |
|---|---:|
| Twelve-row Wave 1 repair | 38087856 |
| Exact Wave 1 repair gate | 38087857 |
| Wave 2 controller | 38087858 |
| Wave 2 array | 38087859 |
| Wave 2 aggregate/gate | 38087860 |
| Wave 3 controller | 38087861 |
| Wave 3 array | 38087862 |
| Final aggregate/report | 38087863 |

The six stale descendants of the failed initial gate were canceled before this graph was created. The replacement graph is recorded in `results/phase6/overnight/timeout_repair_job_graph.json`.

## Validation

- Local complete suite: 65 passed.
- HPG focused amendment/gate tests: 3 passed.
- Temporary dry run against the exact original manifest: preserved 20 rows, amended 12 rows, no duplicate IDs.

## Status command

```bash
ssh hpg 'squeue -j 38087856,38087857,38087858,38087859,38087860,38087861,38087862,38087863 -o "%.18i %.32j %.10T %.10M %R"'
```

Do not combine partial timed-out checkpoints with repaired rows. Only completed row JSON files matching the amended manifest are admitted by the gate.
