# Phase 6 confirmation-v2 timeout repair

## Current execution

The learned-key mechanism rows were empirically slower than the fixed-key rows. Four rows reached Slurm's original `04:00:00` wall limit after their 40M-token checkpoint; the registered endpoint is 50M tokens. This is an execution-limit failure, not a scientific endpoint.

On 2026-07-29 EDT:

- pending original-array tasks were extended to `08:00:00`;
- four running learned-key tasks that Slurm could not extend were canceled before timeout and included in the repair;
- the four prior timeouts and four canceled tasks were restarted from exact immutable manifest rows as repair array `38319264`;
- final report job `38203849` now depends on `afterany:38203848` and `afterok:38319264`.

The repair manifest is `/blue/uf-dsi/rvalle1/KAM_confirmation_v2_results/results/phase6/confirmation_v2/timeout_repair_manifest.jsonl`. It contains original indices `133,134,136,137,139,140,142,143`, has eight rows, and SHA-256 `2449d1296db236e2a22b3fc5bdcd3139f6be98333ace1d9f55941939f7eda9c4`.

## Scientific integrity

This repair changes only the Slurm wall-time ceiling. It does not change row IDs, seeds, corpus/data order, architecture, optimization schedule, precision, validation checkpoints, or the registered 50M-token budget. Partial checkpoints are not admitted as results and are not resumed because they lack the complete optimizer and data-generator state needed for an equivalent continuation.

Future clean submissions use an eight-hour ceiling in `scripts/submit_phase6_confirmation_hpg.sh`. Build an auditable exact subset with:

```bash
python scripts/build_phase6_confirmation_timeout_repair.py \
  --manifest /path/to/manifest.jsonl \
  --indices 133,134,136,137,139,140,142,143 \
  --expected-sha256 7a47d6a54cda5e782f37c1db86081838eab8561b361bac52f57a6c3ba9f851df \
  --output /path/to/timeout_repair_manifest.jsonl \
  --audit-output /path/to/timeout_repair_audit.json
```

## Interpretation boundary

The repair preserves the registered confirmation analysis. However, the mechanism cohort matches total parameter count rather than isolating key learnability at identical depth, width, support count, and expert parameterization. Treat it as a learned-memory lifecycle audit, not a clean causal comparison of fixed versus learned keys. A subsequent mechanism-only experiment should hold those architecture fields constant.
