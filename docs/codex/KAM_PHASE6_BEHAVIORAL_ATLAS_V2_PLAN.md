# Phase 6.2 behavioral atlas and solution-space experiment plan

## Status and decision boundary

Status: **design draft — do not submit**.

This campaign is a pre-registered, staged study of what sparse KAM memory learns,
when its geometry is useful, how much function it can express, why it succeeds or
fails, and which follow-on mechanisms are justified. It is broader than the
Phase 6.1 parameter-dynamics study and must not be used to rescue a failed
confirmation result by searching until a favorable comparison appears.

The immediate decision is whether to replace the planned Phase 6.1 60-row main
with this more complete design. No HPG production array should be submitted until:

1. optimizer labels match executable behavior;
2. the anchor/probe protocol and estimands are implemented and tested;
3. Stage 0 measures tracing overhead and repeatability on an L4;
4. row counts, storage, and runtime forecasts are generated from an immutable
   manifest; and
5. the user approves the final compute envelope.

Machine-readable design: `configs/phase6/behavioral_atlas_v2.yaml`.

## Why the Phase 6.1 pilot is not yet enough

The completed 10-row pilot passed every instrumentation and freeze-integrity
audit. It also exposed useful design limitations:

- The manifest calls the joint optimizer `joint_sgd`, but the runner uses AdamW.
  The new campaign uses executable labels such as `joint_adamw`.
- The two pilot seeds and 5M-token budget are non-inferential.
- The pilot changes only key trainability, freeze timing, and an 8:1 alternating
  schedule on one 10M-parameter architecture and one corpus.
- Checkpoint gradient and update metrics describe one optimizer step. They do not
  estimate the distribution or signal-to-noise ratio of updates inside a window.
- Routing Jaccard compares current queries under current versus initial keys. It
  isolates a key counterfactual but does not separate query drift, key drift, and
  their interaction.
- Support utilization uses at most 256 probe states per layer. With 1,024
  supports, this is too sparse for a stable dead-support estimate.
- Parameter displacement alone cannot show whether memory changes the function,
  increases useful capacity, or merely co-adapts with the backbone.
- TinyStories byte-level next-token loss does not establish retrieval,
  compositional, dynamical, length-generalization, or online-adaptation ability.

Pilot values are therefore hypotheses, not conclusions. At 5M tokens, learned
joint keys moved by roughly 2.6–3.4% relative L2 and routing identity fell to
roughly 0.70–0.75, while the 8:1 alternating keys moved by only about 0.6% and
routing identity remained about 0.91. Support-use entropy stayed near 0.72. This
suggests that parameter motion, routing motion, support coverage, and predictive
benefit must be measured separately.

## Scientific decomposition

The campaign separates five questions that must not be collapsed into one score.

### Q1. Optimization and lifecycle

Do key updates naturally decay, does explicit freezing improve final tuning, and
do joint versus alternating schedules change stability or generalization?

### Q2. Functional use

Does the memory branch causally affect predictions, and are learned keys
responsible for that effect rather than values, gates, backbone drift, or extra
parameters?

### Q3. Expressivity

How much task-relevant function can the memory branch represent as support count,
top-k, expert rank, router metric, architecture scale, and task structure vary?

### Q4. Generalization and adaptation

Does the learned representation extrapolate, survive distribution shift, adapt
prequentially without leakage, and retain pre-shift behavior?

### Q5. Systems value

At equal tokens, how do quality, active computation, total storage, throughput,
VRAM, wall time, and energy trade off? Systems outcomes remain separate from
causal architecture comparisons.

## Two complementary comparison lenses

### Component-causal lens

The decoder shape, active computation, support count, expert form, initialization,
data bytes, data order, token budget, precision, and checkpoint schedule are
identical. A pair differs in one declared mechanism only. This lens supports
claims such as “training keys improved loss” or “freeze at 80% helped.”

### End-to-end utility lens

T0, T-WIDE, T-MEMTOK, T-MOE, T-PKM, fixed KAM, learned KAM, and promoted KAM
variants are compared at registered total-parameter scales and equal token
budgets. Active parameters, measured FLOPs, wall time, throughput, and VRAM are
reported rather than forced to be identical. This lens supports Pareto claims,
not single-component causal claims.

## Immutable data and anchor protocol

- Primary language corpus: the existing hashed TinyStories V2 128 MiB train
  slice with separate validation and held-out test files.
- Cross-corpus replication: Tiny Shakespeare is retained as a small-domain
  stress test and explicitly caveated as a repeated low-diversity corpus.
- Controlled tasks: variable copy with length extrapolation, associative recall
  with distractors, hidden-regime transitions, switching Mackey-Glass, and stable
  switching NARMA.
- Every corpus/task records bytes or generator version, split ranges, tokenizer,
  sample-order seed, and SHA-256.
- Test sets are evaluated once after locked training. Checkpoint selection and
  screening use validation or development streams only.
- Each domain gets an immutable anchor bank containing at least 8,192 token
  states per layer, stratified by task condition where labels exist. Anchor IDs,
  input bytes, targets, and hashes are stored.
- The same anchor inputs are evaluated at every checkpoint. Initial and current
  hidden queries are both retained for the routing decomposition.
- Probe sufficiency is checked by doubling the anchor bank on Stage 0 rows.
  Utilization, functional rank, and intervention effects must change by less than
  their registered tolerance before the standard bank is accepted.

## Formal metric families

The inferential unit is a paired training seed. Tokens, supports, layers,
checkpoints, interventions, and task examples are repeated measurements, never
independent replicates.

### Performance and learning

- Training and validation cross-entropy, perplexity, accuracy where meaningful,
  and final held-out test loss.
- Area under the validation learning curve at registered token checkpoints.
- Tokens and GPU-seconds to reach fixed validation-loss thresholds.
- Validation-to-test generalization gap.
- Calibration error, Brier score where probabilities are comparable, predictive
  entropy, and rare-condition performance.
- For dynamics: prediction, truth, signed error, absolute error, log absolute
  error, one-step MSE, rollout error, and stability/failure rate.
- For retrieval: exact-match accuracy by length, distractor count, delay, and
  required number of associations.

### Parameter motion

For parameter group `g` at checkpoint `t`:

`relative drift = ||theta_g(t) - theta_g(0)||_2 / ||theta_g(0)||_2`

Record cumulative and incremental drift, cosine/angle to initialization,
parameter norm, fraction changed, raw/clipped gradient norm, update norm,
update-to-weight ratio, effective learning rate, and optimizer state norm.

Window-level online accumulators record mean, median, p90, variance, and
signal-to-noise ratio of gradient and update norms. Expensive full snapshots are
reserved for registered checkpoints.

Natural stabilization remains:

`late pre-freeze key update rate / early key update rate`

The practical “nearly frozen” threshold remains 0.25, but the campaign also
reports absolute update-to-weight rates so a ratio cannot hide two tiny or two
large windows.

### Routing and support geometry

- Top-k routing Jaccard, weighted route overlap, route-margin distribution, route
  entropy, global effective support count, support-frequency Gini coefficient,
  dead-support fraction with a seed-level interval and anchor-bank sufficiency check, load-balance error, and
  support-specialization mutual information.
- Key angular displacement, pairwise-distance distribution, nearest-neighbor
  retention, stable rank, participation ratio, normalized spectral entropy,
  minimum separation, and layerwise effective dimension.
- Metrics use the configured router metric; cosine, dot, and negative-L2 routing
  are not evaluated with an implicit dot-product proxy.

Routing drift is decomposed on the same anchor inputs:

| Queries | Keys | Meaning |
|---|---|---|
| `Q0` | `K0` | initial reference |
| `Qt` | `K0` | backbone/query drift only |
| `Q0` | `Kt` | key drift only |
| `Qt` | `Kt` | realized current routing |

The residual beyond the two main effects is reported as query-key interaction.

### Functional motion and expressivity

- Memory contribution ratio:
  `||memory_update||_2 / ||pre_memory_residual||_2`, by layer and checkpoint.
- Alignment of memory update with the final residual and loss gradient.
- Anchor-set logit drift, predictive KL divergence, top-1 flip rate, and branch
  function drift across checkpoints.
- Linear CKA of hidden states and memory outputs across checkpoints.
- Stable rank, participation ratio, and singular-value entropy of the anchor
  memory-output matrix.
- Randomized-SVD estimates of the leading singular spectrum of Jacobians with
  respect to hidden inputs, keys, values/experts, and gates on a fixed subset of
  seeds and checkpoints.
- Hutchinson estimates of Jacobian Frobenius norm and gradient covariance trace.
- Routing-partition complexity: unique top-k signatures and occupancy on fixed
  anchors.
- Capacity curves across parameter scale, support count, top-k, and expert rank.
- Interpolation versus length/composition/delay extrapolation gap.
- Sample efficiency and scaling slope in loss versus tokens, stored parameters,
  and active parameters.

“Higher expressivity” is not inferred from parameter count. It requires a
measured increase in functional rank, task capacity, or extrapolative fit at a
registered resource level.

### Causal interventions

Interventions are evaluated from saved checkpoints without retraining:

- memory branch off / gate forced to zero;
- keys restored to initialization;
- experts/values restored to initialization;
- keys shuffled alone;
- experts shuffled alone;
- keys and experts jointly permuted with the same permutation;
- uniform routing;
- top, random, and bottom support deletion at 1%, 5%, 10%, 25%, and 50%;
- layerwise memory deletion;
- evaluation-time top-k and temperature sweeps;
- checkpoint swaps `K0/V0`, `K0/Vt`, `Kt/V0`, and `Kt/Vt`.

Joint key/expert permutation is a symmetry sanity check and should leave the
function unchanged up to numerical tolerance. Failure blocks interpretation.

For each intervention record change in loss, KL divergence, top-1 flip rate,
logit L2 change, calibration, branch norm, and runtime. Deletion importance is
measured on held-out validation anchors; the test set remains untouched until
the final registered evaluation.

### Adaptation and forgetting

All adaptation uses predict-then-update prequential order. Compare no update,
full-model AdamW, value/expert-only AdamW, geometry-only AdamW, gate-only AdamW,
implemented RLS/NLMS readouts, and trust-region geometry with algebra transport.
Declared and effective adapters must be identical.

Record early/late error, cumulative regret, area under post-shift error, recovery
half-life, adaptation gain, pre-shift retention after adaptation, catastrophic
forgetting, parameter/function drift, and trust-region rejection reasons.

## Staged experiment graph

Stages are decision-gated to avoid spending thousands of GPU-hours on invalid
instrumentation or dominated configurations. A gate may stop later work, but
there is no optional stopping or seed extension within an inferential stage.

### Stage 0 — instrumentation, repeatability, and L4 profile

- Three non-inferential seeds.
- Six canonical arms at 2M tokens: T0, fixed KAM, joint-AdamW learned KAM,
  freeze-80, ALT8 freeze-80, and ALT32 freeze-80.
- Eighteen functional rows plus six paired trace-overhead/repeatability profiles.
- Standard versus doubled anchors on selected rows.
- Eager versus `torch.compile` benchmark on the exact L4 shape.
- Required gates: deterministic manifests, initial-state identity, adapter and
  optimizer provenance, finite metrics, event order, symmetry intervention,
  trace overhead at or below 10% or an explicitly revised cadence, anchor
  sufficiency, restart identity, and bounded storage.

Stage 0 is excluded from all scientific inference.

### Stage 1 — core lifecycle and optimization causality

Primary arms use 30 fresh paired seeds at 50M tokens:

1. exact fixed keys;
2. joint AdamW, freeze at 80%;
3. joint AdamW, no freeze;
4. ALT8 AdamW, freeze at 80%.

Secondary arms use 12 of the same seed IDs:

5. joint freeze at 25%;
6. joint freeze at 50%;
7. ALT32 freeze at 80%;
8. joint cosine geometry-LR decay without a hard freeze.

This is 168 rows. Primary questions are learned geometry versus exact fixed,
freeze-80 versus no-freeze, and ALT8 versus joint freeze-80. Freeze timing and
ALT32 are a Holm-corrected secondary family. The decay arm tests whether smooth
stabilization can replace an abrupt freeze.

### Stage 2 — constrained solution-space screen and confirmation

Use a deterministic constrained D-optimal/space-filling design rather than the
full Cartesian product. Seventy-two cells cover:

- geometry LR: `3e-6, 1e-5, 3e-5, 1e-4`;
- joint, ALT2, ALT8, and ALT32 schedules;
- random, data-sample, k-means, and farthest-point initialization;
- dot, cosine, and negative-L2 routing;
- temperature: `0.5, 1.0, 2.0`;
- supports: `256, 1024, 4096`;
- top-k: `1, 4, 16`;
- vector, low-rank-4, low-rank-16, and shared-basis experts;
- no regularizer versus coverage/repulsion/load-balance regularization;
- trust region off versus on.

Every level must appear at least twelve times. The registered two-factor interactions are schedule × geometry LR, supports × top-k, expert form × supports, initialization × router metric, regularization × supports, and trust region × geometry LR; these and the main effects
must be estimable. Three fresh screening seeds and 10M tokens produce 216 rows.
Screening ranks configurations on a validation-only constrained Pareto rule:
quality, functional branch use, support health, stability, and L4 cost.

At most six non-dominated configurations are frozen before confirmation. Each is
then run for 50M tokens on 12 new paired seeds: at most 72 rows. Confirmation is
fixed-sample; screening seeds are not reused.

### Stage 3 — expressivity and generalization atlas

Screen T0, T-WIDE, T-MEMTOK, T-MOE, T-PKM, fixed KAM, canonical learned KAM,
and one promoted KAM configuration.

- Registered scales: approximately 2M, 10M, and 30M stored parameters.
- Tasks: TinyStories language, variable copy/length extrapolation, associative
  recall with distractors, hidden regimes, and switching dynamics.
- Four screening seeds per architecture/scale/task.
- Equal-token budgets within a task/scale cell; compute is an outcome.
- Architecture promotion is stratified by task. A dynamics loss cannot promote a
  language model and vice versa.

The screen contains up to 480 rows. Before any confirmation data are inspected,
freeze at most four architectures per task and run 12 fresh paired seeds at the
selected scale/budget. Cross-task meta-analysis is secondary and hierarchical;
task examples are never pooled as independent observations.

### Stage 4 — checkpoint reuse and causal behavioral atlas

Reuse Stage 1 and Stage 3 checkpoints. Batch all registered interventions for one
model/checkpoint/anchor bank in a single evaluation job. Deep Jacobian/SVD
diagnostics run on a balanced subset of six seeds at 0%, 20%, 50%, 80%, and 100%
training; lightweight causal interventions run on all primary seeds.

No training is repeated. This stage determines whether parameter motion produces
functional motion, which layer/supports matter, and whether apparent memory gains
survive deletion and swap controls.

### Stage 5 — shift adaptation and limitation tests

Screen on three seeds across four shifts: corpus/domain, sequence length,
hidden-regime transition, and delayed-dynamics transition. Use the registered
adapters above and a small set of Stage 3 models. Freeze the adapter/model matrix
before a 12-seed confirmation.

This stage is allowed to conclude that KAM is useful only for adaptation, only
for a task family, or not useful at the tested scales. It cannot overwrite the
Stage 1 lifecycle estimands.

### Stage 6 — evidence-triggered solution campaign

Stage 6 is a new preregistration, not an automatic search. Observed limitations
map to candidate solutions:

| Observed limitation | Candidate next mechanisms |
|---|---|
| Persistent routing collapse or many truly dead supports | coverage, repulsion, load balance, support revival/birth-death |
| Large key drift without functional gain | geometry LR decay, anchoring, EMA keys, trust region, earlier freeze |
| Useful routing but weak experts | larger/shared expert rank, algebra solve, value-only tuning |
| Key/value co-adaptation breaks after geometry movement | algebra transport, alternating solves, variable projection |
| Branch stays near zero or deletion has no effect | gate warm start, staged gate training, branch auxiliary objective, or remove memory |
| High function drift and forgetting during shift | value-only/RLS adaptation, trust-region geometry, replay/retention constraint |
| Capacity saturates with supports | learned metric, larger top-k, expert-rank scaling, hierarchical/product routing |
| Router dominates L4 time or VRAM | chunked exact routing, validated approximate routing, product-key shortlist |

No solution is promoted merely because it was suggested by the same data used to
evaluate it. It receives fresh seeds and a separate held-out confirmation.

## Locked inferential rules

- Pair architectures by exact training seed, data-order seed, corpus/task hash,
  token budget, and checkpoint target.
- Primary effects use paired log-loss ratios, 20,000-replicate seed bootstrap
  intervals, exact or deterministic Monte Carlo paired sign-flip tests, win rate,
  median paired effect, and standardized paired effect.
- Use Holm correction inside each declared primary or secondary family.
- Use equivalence tests when claiming “no meaningful difference.” The default
  smallest effect of interest is 1% relative loss for component-mechanism
  equivalence and 2% for end-to-end utility.
- Longitudinal claims reduce to seed-level summaries such as area under the
  learning curve, time to threshold, or preregistered early/late window ratios.
- Report median and distributional intervals in addition to means for skewed
  drift, routing, and intervention effects.
- Hierarchical task/domain models are secondary. They include seed and task
  effects and never treat token examples, supports, or checkpoints as replicates.
- Missing scientific rows are not replaced. Infrastructure failures rerun the
  exact content-addressed row only.
- No test-set peeking, optional stopping, seed replacement, unregistered
  cross-architecture calibration, or post-hoc checkpoint selection.
- Negative, null, and equivalence outcomes are reported with the same artifact
  completeness as favorable outcomes.

## Figure contracts

Every figure declares data grain, units, anchor/task scope, paired seed count,
and uncertainty method. Comparable panels use common scales. Color is backed by
line style, marker, direct label, or faceting.

| Figure | Grain and form | Question |
|---|---|---|
| Train/validation learning curves | seed × checkpoint; faceted line with median and paired interval | How quickly and reliably does each model learn? |
| Prediction/truth/error | one locked dynamics task/seed plus seed-level error summary; line and log-error panel | What kind of errors remain, and are they transient or structural? |
| Generalization and calibration | seed dot/interval plus reliability curve | Does improved fit transfer to held-out data? |
| Parameter-group drift | seed × group × checkpoint; small-multiple line | Which groups move, when, and by how much? |
| Gradient/update dynamics | windowed median/p90 and signal-to-noise; log line | Are parameters still receiving meaningful updates? |
| Layer/checkpoint drift | layer × checkpoint heatmap, one scale per metric | Where does adaptation occur? |
| Angular and spectral geometry | paired dots/intervals by checkpoint | Does geometry rotate, collapse, or expand? |
| Stabilization interval | seed ratio dot/interval with 0.25 reference | Is “nearly frozen” statistically supported? |
| Query/key routing decomposition | four-state heatmap/small multiples | Is routing change caused by backbone queries, keys, or interaction? |
| Support usage and inequality | occupancy Lorenz curve plus entropy/Gini/dead-support interval | Is memory capacity broadly used? |
| Route margin and churn | distribution/interval by checkpoint | Is routing stable or boundary-sensitive? |
| Branch contribution and function drift | layer/checkpoint lines | Does the memory branch materially alter computation? |
| CKA and functional rank | checkpoint matrix and singular-spectrum interval | How does representational capacity evolve? |
| Causal intervention effects | seed-level forest and deletion dose-response | Which memory components causally support predictions? |
| Capacity/scaling curves | loss or functional rank versus parameters/tokens, faceted by task | Where does KAM add expressivity or saturate? |
| Resource-quality Pareto | seed/config scatter with active parameters and VRAM context | Is any gain worth its systems cost? |
| Adaptation/recovery | prequential error and paired regret/forgetting intervals | Can the model adapt without erasing prior ability? |

## L4 execution and storage optimization

1. Profile 200 steady-state steps on an actual L4 before locking row forecasts.
2. Retain bf16 autocast, TF32, fused AdamW, nonblocking transfers,
   `zero_grad(set_to_none=True)`, and inactive-group gradient disabling.
3. Benchmark eager versus `torch.compile`; enable compile only if numerical
   agreement passes and steady-state throughput improves by at least 10% after
   amortizing compile time.
4. Pre-generate and hash data start indices. Use pinned host buffers or a
   deterministic GPU-resident index path when it improves throughput.
5. Accumulate cheap gradient/update statistics online. Clone all parameters only
   at registered checkpoints and run Jacobian/SVD diagnostics in separate
   checkpoint-reuse jobs.
6. Use exact routing for scientific comparisons. Chunk only to control VRAM.
   Approximate routers must report exact-router recall and may not silently
   replace exact routing.
7. Bucket Slurm arrays by predicted runtime, scale, and memory so short rows are
   not trapped behind long rows. Keep one L4 per row and `%4` occupancy.
8. Use content-addressed rows, atomic result writes, resumable registered
   checkpoints, and `afterany` report jobs that record failures without promoting
   incomplete stages.
9. Batch all interventions for one loaded checkpoint. Do not reload or retrain a
   model for each ablation.
10. Save float16 key snapshots at all dynamics checkpoints, full model/optimizer
    state only at registered restart/deep-analysis checkpoints, and compressed
    seed-grain tables for reports.
11. Forecast checkpoint, trace, report, and temporary storage before submission;
    reserve at least 25% headroom.
12. Record GPU model, driver, CUDA, PyTorch, precision, compile mode, selected
    kernels, peak allocated/reserved VRAM, tokens/s, wall time, estimated FLOPs,
    and sampled power/energy when available.

At the observed Phase 6.1 pilot rate, a 50M-token 10M-parameter row is roughly
four L4 GPU-hours. Stage 1 alone is therefore approximately 670–720 GPU-hours,
or seven to eight days at uninterrupted four-way occupancy. The complete
solution-space campaign can exceed 2,000 GPU-hours. It should be deployed one
validated stage at a time, not as one monolithic queue.

## Required artifacts per stage

- immutable manifest, SHA-256, git commit, environment, dataset, anchor, and
  initial-state hashes;
- planned versus resolved token budgets and exact sample-order provenance;
- one result JSON and compact trace table per row;
- seed-grain metrics and paired-comparison tables in Parquet plus JSONL fallback;
- registered checkpoints and key snapshots;
- audit and failure ledger;
- figure map and all required PNG/SVG figures;
- concise human report;
- LLM handoff with decisions, caveats, exact paths, and evidence-triggered next
  actions.

## Implementation order

1. Correct optimizer/provenance labels in the Phase 6.1 code without changing the
   mathematical update.
2. Implement immutable anchor banks and query/key routing decomposition.
3. Add online window accumulators, functional branch metrics, CKA/rank metrics,
   and permutation symmetry checks.
4. Implement Stage 0 manifest, storage/runtime estimator, L4 profiler, and audit.
5. Run Stage 0 only.
6. Lock Stage 1 row counts and estimands after validating measurement quality,
   without inspecting any Stage 1 outcome.
7. Submit Stage 1 from a clean committed checkout.
8. Build later stage manifests only after their registered predecessor gate.

## Stage 0 implementation and L4 profile gate (2026-07-30)

Status: original Stage 0 finished with 23/24 rows and preserved decision `STAGE0_BLOCKED`. Array `38373854` completed except compile index 23; report job `38373855` completed and recorded the blocked gate. The repair and Stage 1 dependency graph is documented in `KAM_PHASE6_BEHAVIORAL_ATLAS_STAGE0_REPAIR_STAGE1_HANDOFF.md`.

### Implemented contracts

- 24 deterministic rows: six functional arms over three paired seeds plus six profiling variants.
- Ten registered training checkpoints support median-and-seed-range learning curves and time-resolved drift, gradient/update, prediction, routing, utilization, and contribution plots.
- Executable AdamW provenance, immutable hashed anchors and sample orders, exact query/key routing decomposition, support utilization, parameter/gradient/update windows, freeze integrity, functional contribution, CKA/rank/spectral metrics, prediction drift, restart identity, and matched key/expert permutation audits.
- Atomic row outputs, audits, forecasts, and eight PNG/SVG figure pairs. HPG Stage 0 submission is gated on the locked L4 profile and a clean committed checkout.

### Validation evidence

- Local: 95/95 tests passed; a 65,536-token RTX 4070 BF16 CUDA smoke passed.
- HPG revision 1 (`38371497_0`/`38371498`) is preserved as `L4_PROFILE_BLOCKED`: BF16 numerical drift was incorrectly used as the semantic permutation gate.
- Revision 2 (`38372064_0`/`38372065`) separated exact semantic symmetry from deployed-precision diagnostics and passed.
- Final locked revision 3 (`38372382_0`/`38372383`) additionally disabled TF32 only during the FP32 semantic check and made BF16 operational stability explicit. Every audit passed, both error logs were empty, and the decision is `L4_PROFILE_PASS`.

### Locked profile observations

One noninferential row with a 250k-token target realized 251,904 tokens in 83.84 seconds on one NVIDIA L4 (3,004.6 tokens/s, 600.0 MiB peak allocated VRAM, 3.20 GiB Slurm MaxRSS). Validation loss moved from 5.685 to 2.504 and test loss was 2.626.

Memory-key relative drift reached 0.00413 before the 80% freeze and remained exactly unchanged afterward; post-freeze gradients and optimizer updates were zero. Strict FP32 matched-permutation logit difference was exactly zero. BF16 max difference was 0.03125, top-1 flip rate 0.195%, and predictive KL 9.53e-6, all within the operational audit.

Final anchor support entropy was 0.713, dead-support fraction 0.467, effective support count 107.9/1024, routing Jaccard to initialization 0.164, memory contribution ratio 0.00240, and memory-output stable rank 3.92. These describe an early single profile and are not architecture comparisons.

### Evidence and next gate

- Final report and figures: [`reports/phase6/behavioral_atlas_v2/l4_profile_r3/`](../../reports/phase6/behavioral_atlas_v2/l4_profile_r3/)
- Final summary, manifest, row, and trace: [`results/phase6/behavioral_atlas_v2/l4_profile_r3/`](../../results/phase6/behavioral_atlas_v2/l4_profile_r3/)
- Preserved blocked revision: [`results/phase6/behavioral_atlas_v2/l4_profile/`](../../results/phase6/behavioral_atlas_v2/l4_profile/)

The [locked 24-row manifest](../../configs/phase6/behavioral_atlas_v2_stage0_manifest.jsonl) SHA-256 is `9514749acc0c5ac3432569d48c6157bd8d4c1a617cfa7b06b170c3cd005bf78a`; forecast is 4.224 L4 GPU-hours, about 1.06 hours at four-way occupancy, and 1.82 GiB storage including 25% headroom. The signed query/key interaction value is an inclusion-exclusion residual: negative values mean combined churn is less than the sum of isolated query and key churn, not negative route change.

Original Stage 0 completed 23/24 rows and remains preserved as `STAGE0_BLOCKED`; the compile candidate failed in the PyTorch 2.8 CUDA Graph allocator. The revisioned noninferential repair uses prediction-behavior BF16 tolerances, fixed-bank anchor invariants, a 16,384-to-32,768 anchor reevaluation from a saved checkpoint, and a no-CUDA-graph compile candidate. Stage 1 is now fully implemented as the locked 168-row paired design and is dependency-gated on `STAGE0_REPAIRED_PASS`. See `KAM_PHASE6_BEHAVIORAL_ATLAS_STAGE0_REPAIR_STAGE1_HANDOFF.md`.
