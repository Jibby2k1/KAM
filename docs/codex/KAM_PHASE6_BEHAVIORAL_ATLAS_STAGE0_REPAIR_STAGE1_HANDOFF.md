# Phase 6.2 Stage 0 repair and Stage 1 deployment handoff

## Current decision

The original noninferential Stage 0 evidence is preserved and remains `STAGE0_BLOCKED`: 23/24 rows completed, while the `torch.compile(mode="reduce-overhead")` candidate failed in PyTorch 2.8's CUDA Graph allocator. The successful rows passed strict FP32 permutation identity, repeatability, freeze integrity, trace overhead, optimizer provenance, and restart identity.

This revision implements a two-row measurement repair and a dependency-gated 168-row Stage 1 deployment. Stage 1 may start only if the revisioned repair report returns `STAGE0_REPAIRED_PASS`.

### Deployment repair ledger

Repair revision 1 jobs `38384977`–`38384981` exposed two infrastructure assumptions before Stage 1 began: regenerated Stage 0 IDs did not match the original immutable manifest, and the clean checkout did not contain git-ignored corpus files. Both repair rows recorded explicit failures and the dependency gate held Stage 1. Revision 2 reads row IDs directly from the original manifest SHA and links the checksum-validated corpora from the prior Stage 0 checkout. Revision-1 artifacts remain preserved; only its never-runnable downstream jobs are cancelled.

## Why the measurement rules changed

These changes are confined to noninferential Stage 0 calibration; they do not alter or reinterpret a scientific outcome.

1. **Permutation semantics:** strict FP32 matched key/expert permutation remains the exact semantic invariant.
2. **BF16 operation:** the BF16 gate now uses top-1 prediction changes at or below 2% and mean predictive KL at or below `1e-3`. Maximum absolute logit difference remains reported but is scale-dependent and no longer serves as the gate.
3. **Anchor sufficiency:** the registered Stage 1 anchor contains 16,384 token states and is checked against 32,768 states. Contribution, stable rank, participation ratio, normalized support entropy, and effective support count must change by no more than 5%.
4. **Dead supports:** dead-support fraction remains a fixed-bank descriptive metric. It is not a bank-doubling invariant because unseen occupancy mechanically decreases with additional samples; the repair reports its layer range at both bank sizes.
5. **Compilation:** the repair benchmarks `torch.compile(mode="default")` with CUDA graphs explicitly disabled. Compilation is optional systems evidence. Stage 1 is locked to eager execution for this inferential manifest, avoiding a conditional execution-policy change after the manifest is sealed.

## Registered repair

- Anchor reevaluation loads the completed `learned_joint_freeze80`, seed 76001, 2M-token checkpoint; it does not retrain the successful science row.
- Compile reevaluation reruns only the failed profile row using the stable no-CUDA-graph path.
- Original Stage 0 manifest SHA-256: `9514749acc0c5ac3432569d48c6157bd8d4c1a617cfa7b06b170c3cd005bf78a`.
- Repair implementation: `kam/phase6/behavioral_atlas_repair.py`.
- Repair outputs: `results/phase6/behavioral_atlas_v2/stage0_measurement_repair_r2/` on the configured HPG result root.
- Repair report: `reports/phase6/behavioral_atlas_v2/stage0_measurement_repair_r2/BEHAVIORAL_ATLAS_STAGE0_REPAIR_REPORT.md`.

## Registered Stage 1

Stage 1 is the locked 168-row, 50M-token paired lifecycle study:

- 30 fresh paired seeds, 76101–76130, for fixed keys, joint freeze-80, joint no-freeze, and ALT8 freeze-80: 120 rows.
- The first 12 paired seeds for joint freeze-25, joint freeze-50, ALT32 freeze-80, and joint cosine geometry-LR decay without hard freeze: 48 rows.
- Nine checkpoints: 0, 1M, 2M, 5M, 10M, 20M, 30M, 40M, and 50M tokens.
- One NVIDIA L4 per row, four-way array occupancy, eight-hour per-row limit.
- Projected compute: 670–720 L4 GPU-hours, roughly seven to eight days at uninterrupted four-way occupancy.

Primary comparisons are no-freeze versus fixed, freeze-80 versus fixed, freeze-80 versus no-freeze, and ALT8 versus joint freeze-80. Secondary comparisons assess freeze timing, ALT32, and smooth cosine stabilization. Both families use paired log-loss ratios, geometric relative changes, win rates, and Holm correction.

## Reports and figures

`kam/phase6/behavioral_atlas_stage1_analysis.py` writes:

- a human-facing Stage 1 report;
- a concise LLM handoff;
- seed-grain JSONL and paired-comparison JSONL;
- paired bootstrap intervals, exact or 200,000-draw sign-flip tests, standardized paired effects, and Holm-adjusted p-values;
- learning curves, key-drift curves, the geometry-LR trajectory, paired-effect intervals, and held-out test-loss distributions in PNG and SVG.

## Execution graph

`scripts/submit_phase6_behavioral_atlas_repair_stage1_hpg.sh --submit` creates:

1. a two-row L4 repair array;
2. an `afterany` repair audit;
3. a 250k-token L4 preflight of the cosine arm with `afterok` dependency on the repair audit;
4. a 168-row Stage 1 L4 array with `afterok` dependency on the GPU preflight;
5. an `afterany` Stage 1 report job.

The graph, manifest hashes, clean git commit, wall limits, and job IDs are written to `repair_stage1_job_graph.json`. A failed repair audit prevents Stage 1 GPU work from starting.

## LLM review order

1. This handoff.
2. `docs/codex/KAM_PHASE6_BEHAVIORAL_ATLAS_V2_PLAN.md`.
3. `configs/phase6/behavioral_atlas_v2.yaml` and the generated immutable manifests.
4. The preserved original Stage 0 report.
5. The revisioned repair report and `stage0_repair_summary.json`.
6. After completion, the Stage 1 human report, LLM handoff, paired-comparison table, and figures.

Do not infer model superiority from Stage 0 or its repair. Do not extend Stage 1 seeds or stop it early based on interim outcomes.
