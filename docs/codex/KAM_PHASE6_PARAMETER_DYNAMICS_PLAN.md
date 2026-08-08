# Phase 6.1 matched parameter-dynamics experiment plan

## Decision and boundary

This round asks how learned persistent-memory parameters move, whether that movement naturally stabilizes, and whether freezing geometry for final tuning improves held-out performance. It is a mechanism study. It does not reopen or rescue the failed TinyStories superiority gate from confirmation v2.

Do not submit the main array until the instrumentation pilot passes. The machine-readable contract is `configs/phase6/parameter_dynamics_v1.yaml`.

## Implementation and current status

The pipeline is implemented and its non-inferential pilot was submitted on 2026-07-30:

- isolated HPG checkout: `/blue/uf-dsi/rvalle1/KAM_parameter_dynamics_v1_20260730`;
- 10-row pilot array: `38356889` (`0-9%4`, one L4 per row);
- dependent pilot report: `38356890`;
- manifest SHA-256: `0abd9dcb4bd318c9e56abb63179131e8e702ce120feb904fdbe0c2a438d0807c`;
- the 60-row main study is not submitted.

Implementation map:

- manifest: `kam/phase6/parameter_dynamics_manifest.py`;
- matched runner: `kam/phase6/parameter_dynamics_runner.py`;
- compact trace: `kam/phase6/parameter_trace.py`;
- paired statistics: `kam/phase6/parameter_dynamics_statistics.py`;
- audit/report/figures: `kam/phase6/parameter_dynamics_analysis.py`;
- HPG entrypoint: `scripts/submit_phase6_parameter_dynamics_hpg.sh`.

The L4 path uses bf16 autocast, TF32 matrix multiplication, fused AdamW, `zero_grad(set_to_none=True)`, inactive-group gradient disabling during ALT8, nonblocking host-to-device transfers, expandable CUDA segments, and expensive trace comparisons only at registered checkpoints.

Validation completed before pilot submission:

- full local suite: 86 tests passed;
- local CUDA smoke on an RTX 4070 SUPER: bf16/TF32/fused AdamW passed;
- actual HPG NVIDIA L4 smoke job `38356795`: passed in 22 seconds with an empty error log, 590 MB peak allocated VRAM, exact post-freeze key hash, zero drift, and no post-freeze key gradient.

## Correction required before new science

The confirmation-v2 report says 0/16 learned-memory rows passed the lifecycle audit. That aggregate is misleading. Direct inspection shows:

- 16/16 updated geometry;
- 16/16 had nonzero pre-freeze key gradients;
- 16/16 froze at exactly 80% of tokens;
- 16/16 remained frozen during final tuning;
- 16/16 had exactly zero post-freeze geometry drift.

Only `postfreeze_checkpoint_observed` failed. The runner recorded the 40M checkpoint at the end of an update and froze geometry at the start of the next loop, both with `tokens_seen == 40,001,536`. The audit used `tokens >= freeze_tokens`, so it classified the pre-freeze checkpoint as post-freeze.

The prospective fix is event-based:

1. Record `phase = pre_freeze`, `freeze_event`, or `post_freeze`.
2. Emit a checkpoint immediately after toggling `requires_grad=False`.
3. Use explicit phase for audit membership, with `tokens > freeze_tokens` only as a compatibility fallback.
4. Require unchanged parameter hashes and no key-gradient tensor after freeze.

The locked fixed-key promotion decision remains `RETAIN_AS_DIAGNOSTIC_ONLY` because its primary TinyStories gate failed independently.

## Clean causal design

All arms use the exact T-KAM-L shape selected in confirmation v2: `d_model=104`, 8 heads, 8 decoder/memory layers, `d_ff=416`, 1,024 supports, top-4 routing, rank-4 low-rank experts, and the same parameter budget. Fixed versus learned arms differ only in key trainability and optimizer schedule. They share the same initialized tensor values and data order within every seed.

This removes the earlier confound: T-KAM-F used a shallower/wider decoder, 4,096 supports, and vector experts, while T-KAM-L/ALT used a deeper/narrower decoder, 1,024 supports, and low-rank experts.

| Arm | Keys train until | Optimizer | Question |
|---|---:|---|---|
| Fixed keys | 0% | Algebra only | Exact structural control |
| Joint freeze-50 | 50% | Joint AdamW | Does earlier freeze help? |
| Joint freeze-80 | 80% | Joint AdamW | Primary lifecycle arm |
| ALT8 freeze-80 | 80% | 8 algebra : 1 geometry | Does optimizer separation help? |
| Joint no-freeze | 100% | Joint AdamW | Is final-tuning freeze beneficial? |

Parameter matching may use nonexecuted padding only. It may not change decoder depth, width, support count, expert form, or active computation between arms.

## Staged sample and compute plan

### Instrumentation pilot

- Two fresh paired seeds (`74001–74002`), all five arms.
- 5M tokens per row; 10 rows total.
- Excluded from all inference.
- Must prove exact initial-state identity, complete finite traces, correct freeze-event ordering, unchanged fixed keys, bounded storage, and successful rendering of every required figure.

Any instrumentation repair invalidates and repeats the entire pilot. Main seeds must remain unseen.

### Fixed-sample main study

- Twelve fresh paired seeds (`74101–74112`), all five arms.
- TinyStories V2 128 MiB, identical immutable train/validation/test bytes.
- 50M tokens per row and one final held-out test evaluation.
- Checkpoints at 0, 1M, 2M, 5M, 10M, 20M, 30M, 40M, and 50M tokens.
- 60 rows, approximately 240–258 L4 GPU-hours or 60–65 hours at four-way occupancy.
- Eight-hour Slurm limit per row.

There is no optional stopping, seed replacement, or post-hoc extension.

## Parameter trace

At every checkpoint, record these groups separately:

- memory keys;
- memory experts/values;
- memory gates;
- attention;
- feed-forward blocks;
- embeddings;
- output head.

For each group, record parameter norm, cumulative and incremental displacement, relative displacement, cosine similarity to initialization, raw and clipped gradient norms, update-to-weight ratio, and changed-element fraction. Capture gradient and update quantities immediately before and after `optimizer.step`; checkpoint-time residual gradients are not an acceptable substitute.

Memory-specific summaries include key angular displacement, effective rank, spectral entropy, nearest-neighbor retention, top-k routing Jaccard against initial routing, support-use entropy, dead-support fraction, and gate scale. Store full model checkpoints only at 0/40M/50M. Store float16 key snapshots at every registered checkpoint. Inferential units are seeds, never individual keys, supports, layers, or tokens.

## Locked questions and statistics

### Natural stabilization

Define normalized key update rate as incremental relative key displacement per million tokens. For learned arms, compare the 30–40M pre-freeze window with the 1–10M window:

`stabilization ratio = late pre-freeze update rate / early update rate`.

A “nearly frozen before explicit freeze” claim requires the upper paired-bootstrap 95% bound to be below `0.25`. Merely showing a downward curve is not enough.

### Does learned geometry help before freeze?

At 40M validation tokens, compare joint freeze-80 and ALT8 freeze-80 with the exact fixed-key control using paired log-loss ratios. Apply Holm correction to the two comparisons.

### Does explicit final tuning help?

At 50M held-out test, compare joint freeze-80 with joint no-freeze. Report the paired log-loss ratio, bootstrap interval, exact paired sign-flip p-value, win rate, and standardized paired effect.

### Freeze integrity

Every frozen row must have an unchanged key hash, relative post-freeze drift at or below `1e-12`, and no allocated key-gradient tensor after the freeze event.

Parameter-change/performance association is secondary and descriptive. Report a seed-cluster bootstrap interval; do not treat layers or supports as independent observations.

## Figure contracts

All figures use tokens on the horizontal axis, a neutral background, one color per optimizer arm, line style/markers in addition to color, a visible freeze event, and median plus paired-seed 95% intervals where applicable.

| Figure | Analytical question | Form and fields | Supported takeaway |
|---|---|---|---|
| Parameter-group relative drift | Which model components move? | Faceted line; tokens × relative L2 displacement; facet parameter group | Compares movement scale and timing across groups |
| Key update rate and freeze | Do keys naturally stabilize? | Log-scale line; tokens × relative update per 1M tokens; explicit freeze marker | Shows decay versus abrupt forced freeze |
| Layer/checkpoint drift | Where does key movement occur? | Heatmap; layer × checkpoint, log relative displacement; facet arm | Reveals layer-localized adaptation or collapse |
| Key angular displacement | How far do key directions rotate? | Seed-level box/dot intervals at 10M/20M/40M/50M | Separates directional movement from norm growth |
| Routing stability and usage | Does movement alter retrieval behavior? | Two-panel line/dot: top-k Jaccard and support entropy | Connects parameter motion to functional routing |
| Stabilization ratio | Is “nearly frozen” statistically supported? | Dot-and-interval by learned arm with 0.25 reference | Primary stabilization decision visual |
| Parameter change vs validation change | Is more movement associated with benefit? | Scatter at seed-arm grain; relative key drift × paired validation improvement | Descriptive coupling, outliers, arm structure |
| Final-tuning freeze effect | Does freeze improve generalization? | Paired slope plus effect interval for freeze-80 vs no-freeze | Direct final-tuning counterfactual |

Do not use a raw “parameter value over time” plot: millions of coordinates are unreadable and parameter norms alone cannot distinguish rotation from useful adaptation. The planned figures combine displacement, direction, gradients, routing behavior, and performance.

## Required outputs

- Immutable manifest and SHA-256.
- Environment, commit, Slurm graph, dataset, and initial-state hashes.
- One JSON result and one compact parameter-trace table per row.
- Registered key snapshots and 0/40M/50M model checkpoints.
- Seed-level metrics and comparison Parquet files.
- Eight figures above in PNG and SVG.
- Concise human report.
- LLM handoff containing decisions, caveats, exact paths, and next actions.

## Interpretation rules

- A stabilization pass does not imply better loss.
- Better loss does not imply keys caused the improvement unless the exact matched fixed control is beaten.
- A freeze benefit does not imply natural stabilization.
- The study cannot promote T-KAM-F or overturn confirmation v2.
- If fixed and learned arms do not share exact executable architecture and initialization hashes, block causal interpretation.
