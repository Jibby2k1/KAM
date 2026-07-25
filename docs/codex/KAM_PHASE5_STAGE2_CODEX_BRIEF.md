# KAM Phase V Stage 2 — Codex Execution Brief

## Repository and purpose

Repository: `Jibby2k1/KAM`

Place this file at:

```text
docs/codex/KAM_PHASE5_STAGE2_CODEX_BRIEF.md
```

This document is the authoritative specification for the next KAM experiment stage.

The Stage 1 pilot passed all implementation-validity checks and completed 144/144 runs. Its main signal is:

- the no-memory baseline `D0` remains strongest at approximately 250k active parameters;
- at approximately 1M active parameters, `RF-KV` and `KC-LV` become competitive with or slightly better than `D0`;
- the fully learned bank `DD-L` does not currently outperform `D0`;
- genuinely fixed feature maps (`RF-FULL`, `RFF`) perform poorly.

The next stage must determine whether useful performance comes from:

1. fixed anchors inside a learned coordinate system;
2. learned values rather than learned support locations;
3. data-informed support placement;
4. route-feature dimensionality;
5. a capacity crossover near 1M parameters;
6. task recurrence or finite-memory structure;
7. or statistical noise in the Stage 1 pilot.

Do not begin scaling, online-adaptation confirmation, or natural-language experiments until this stage is complete.

---

# 1. Primary scientific questions

## Q1 — Do learned support locations help?

Compare:

```text
DD-L
RK-LV
KC-LV
RF-KV
```

where:

- `DD-L`: learned keys, learned values, learned coordinate path;
- `RK-LV`: fixed random keys, learned values, learned coordinate path;
- `KC-LV`: fixed data-derived centers, learned values, learned coordinate path;
- `RF-KV`: fixed random keys and fixed random values, learned coordinate path.

A learned-support claim requires `DD-L` to outperform the strongest fixed-key alternative.

## Q2 — Do support values help?

Compare:

```text
DD-L
LK-RV
RK-LV
routes-only controls
```

where:

- `LK-RV`: learned keys, fixed random values;
- `routes-only`: no retrieved-value residual; projected routing features only.

This isolates whether the bank helps through:

- the geometry of its routing activations;
- retrieved value vectors;
- or both.

## Q3 — Which learned component makes random anchors useful?

Starting from `RF-KV`, progressively freeze:

```text
RF-KV-QF     freeze memory-query projection
RF-KV-SF     freeze score/temperature parameters
RF-KV-OF     freeze memory-output projection
RF-KV-BF     freeze the sequence backbone
RF-FULL      freeze the entire feature pathway
```

The experiment must identify the smallest learned subsystem required for `RF-KV` to remain competitive.

## Q4 — Is there a real capacity crossover?

The pilot suggests all banks hurt near 250k parameters but may become useful near 1M.

Test active parameter targets:

```text
250k
400k
600k
800k
1M
1.5M
2M
4M
```

Do not bundle support count, context length, training samples, or optimization budget into the scale label.

Estimate the crossover point where:

\[
\Delta(P)
=
\operatorname{NMSE}_{D0}(P)
-
\operatorname{NMSE}_{bank}(P)
\]

changes sign.

## Q5 — Is the effect task dependent?

The Stage 1 task labels shared one controlled execution path. Stage 2 must use genuinely distinct task generators:

```text
controlled prototype regimes
switching Mackey–Glass
switching NARMA
controlled symbolic regime language
```

No pooled “four-task” interpretation is allowed until the generators are verified to be mechanistically distinct.

---

# 2. Required implementation work

## 2.1 Model factory additions

Modify:

```text
kam/factory.py
kam/model.py
```

Add exact labels:

```text
D0
DD-L
RF-KV
RF-FULL
RK-LV
LK-RV
KC-LV
RFF
DD-A       routes only
DD-V       retrieved values only
DD-B       routes plus retrieved values
```

Keep backward compatibility with historical names.

Every run must record:

```text
key_trainable
value_trainable
query_path_trainable
score_path_trainable
memory_output_trainable
backbone_trainable
route_mode
center_initialization
```

## 2.2 Fixed route dimension

Use projected routes for every primary comparison:

```yaml
route_features: projected
route_projection_dim: 64
```

The final prediction head must receive the same total input dimension across variants.

Raw support activations may be exported for diagnostics but must not create an unmatched readout.

## 2.3 Active-capacity matching

Primary comparisons must satisfy:

```text
absolute active-parameter error <= 1%
padding_parameter_count == 0
```

Record:

```text
active_parameter_count
total_parameter_count
padding_parameter_count
estimated_forward_flops
measured_training_step_ms
```

When exact matching is impossible, mark the row invalid for primary inference rather than silently padding.

## 2.4 Support initializations

Implement:

```text
random_normal
sampled_training_points
kmeans
farthest_point
oracle_regime_centers   # synthetic prototype task only
```

For each initialization, save the selected centers and the exact data subset used.

## 2.5 True task-specific generators

Create or complete:

```text
kam/data/controlled_prototype.py
kam/data/controlled_mackey_glass.py
kam/data/controlled_narma.py
kam/data/controlled_symbolic_regime.py
```

Each generator must produce independent train, validation, test, and ordered prequential streams.

Every generated stream must save:

```text
stream_seed
schedule
regime_boundaries
regime_parameters
noise parameters
observability mode
true memory horizon
context length
return probability
transition type
```

## 2.6 Correct evaluation

Use validation loss only for checkpoint selection.

Before held-out testing:

1. reload the saved best checkpoint;
2. evaluate the full held-out set;
3. compute global MSE, NMSE, NRMSE, MAE, bias, and tail errors;
4. report final-checkpoint results separately.

Global NMSE must be computed as:

\[
\operatorname{NMSE}
=
\frac{
\sum_i(\hat y_i-y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
\]

Do not average batch-level NMSE ratios.

---

# 3. Controlled factors

The Stage 2 search must vary the following factors independently.

## 3.1 Recurrence

```text
return_probability:
0.00
0.25
0.50
0.75
1.00
```

For ordered streams, include:

```text
A -> B -> C
A -> B -> A
A -> B -> A -> C -> A
```

## 3.2 Regime separation

```text
low
medium
high
```

Separation must not alter the number of returns, dwell duration, or noise.

## 3.3 Observability

```text
full
partial
hidden_driver
```

Examples:

- NARMA driver visible versus hidden;
- complete versus aliased Mackey–Glass history;
- explicit versus latent symbolic regime state.

## 3.4 Noise

Vary separately:

```text
observation_noise: none, low, high
process_noise: none, low
driver_noise: fixed by task, independently configured
```

Never use one generic `noise_std` field to represent all three concepts.

## 3.5 Memory sufficiency

Use:

```text
context / true memory:
0.5x
1x
2x
4x
```

This tests whether a bank helps only when the context encoder is underspecified.

## 3.6 Feature-bank size

Sweep:

```text
supports/features:
16
32
64
128
256
```

Keep route projection dimension fixed at 64.

---

# 4. Experiment stages

## Stage 2A — Component ablation

Purpose: determine what makes `RF-KV` useful.

Tasks:

```text
controlled prototype
switching Mackey–Glass
switching NARMA
```

Models:

```text
D0
DD-L
RF-KV
RF-FULL
RK-LV
LK-RV
KC-LV
DD-A
DD-V
DD-B
```

Scales:

```text
600k
1M
2M
```

Seeds:

```text
5 paired training seeds
5 held-out streams per trained model
```

Use medium recurrence and medium separation.

Expected volume:

```text
approximately 450 model runs
plus held-out evaluations
```

Primary outputs:

- learned-key effect;
- learned-value effect;
- learned-coordinate-path effect;
- route-versus-value effect.

## Stage 2B — Capacity crossover

Models:

```text
D0
DD-L
RF-KV
KC-LV
```

Scales:

```text
250k, 400k, 600k, 800k, 1M, 1.5M, 2M, 4M
```

Seeds:

```text
5 paired seeds
```

Tasks:

```text
controlled prototype
switching Mackey–Glass
switching NARMA
```

Use one locked reference condition per task.

Expected volume:

```text
480 runs
```

Fit a smooth effect curve versus log active parameters and estimate uncertainty in the zero crossing.

## Stage 2C — Orthogonal factorial screen

Shortlist no more than:

```text
D0
DD-L
RF-KV
KC-LV
best component ablation from Stage 2A
```

Use the checked-in mixed-level Taguchi L18 design over:

```text
return probability
separation
observability
observation noise
process noise
support count
center initialization
```

Training fidelity:

```text
100%
```

Fidelity is fixed rather than assigned from the design index, so training
budget is not confounded with a scientific factor.

Seeds:

```text
4 paired seeds
```

Expected volume:

```text
1,080 completed trials
```

Pair variants on the same generated streams.

Do not promote configurations solely from the Optuna objective. Require held-out paired effects.

## Stage 2D — Controlled symbolic language replication

Models:

```text
D0
DD-L
RF-KV
KC-LV
```

Factors:

```text
transition entropy
emission overlap
return probability
explicit regime token
context length
```

Use:

```text
5 paired seeds
at least 5 held-out streams per seed
```

The goal is not natural-language perplexity. The goal is to determine whether the fixed-anchor mechanism generalizes from continuous dynamical systems to a controlled discrete language.

---

# 5. Training budgets

For Stage 2A and 2B:

```text
minimum evaluation checkpoints: 50
no early stopping before 40% of budget
save best and final checkpoints
```

Suggested observation/token budgets:

```text
250k–600k parameters: 5M and 15M samples
800k–1.5M:          10M and 30M samples
2M:                 20M and 50M samples
4M:                 30M and 80M samples
```

Run sample-matched and GPU-hour-matched summaries.

A model that needs substantially more compute for the same quality must report that cost as part of the conclusion.

---

# 6. Required diagnostics

For every bank model:

```text
route entropy
effective support count
dead-support fraction
duplicate-support fraction
support-use Gini coefficient
feature Gram effective rank
feature Gram condition number
nearest observed contexts
within-support target variance
within-support local-linear residual
key/value drift
query-path drift
top/random/bottom deletion curves
```

For cross-seed stability:

1. evaluate support activation fingerprints on a common anchor set;
2. match supports with the Hungarian algorithm;
3. report matched fingerprint correlation and nearest-context overlap.

For `RF-KV`, also measure how much the learned query path changes the distribution of distances to fixed keys.

---

# 7. Statistical analysis

The training seed is the inferential unit.

Aggregate held-out streams within seed before inference.

For each preregistered comparison:

- paired mean relative improvement;
- paired bootstrap 95% confidence interval;
- exact paired permutation test;
- standardized paired effect size;
- ±5% equivalence test.

Correct across the small set of declared primary comparisons using Holm adjustment.

Do not treat multiple streams from one checkpoint as independent training replicates.

---

# 8. Decision gates

## Promote learned supports

Require all:

```text
DD-L beats RF-KV and KC-LV by >= 5%
95% paired CI excludes zero
causal deletion indicates learned support use
support behavior is stable across seeds
latency and parameter overhead are justified
```

## Promote fixed-anchor learned geometry

Require all:

```text
RF-KV or KC-LV beats D0 by >= 5%
95% paired CI excludes zero
effect reproduces on at least two distinct generators
RF-FULL remains materially worse
learned coordinate-path ablation identifies the required component
```

## Promote data-centered kernel basis

Require:

```text
KC-LV beats RF-KV
or reaches equivalent quality with fewer supports / less compute
```

## Reject support-location learning

Conclude that learned support locations are unnecessary if:

```text
DD-L is equivalent to fixed-key alternatives within ±5%
and has no compression, adaptation, stability, or interpretability advantage
```

## Stop the bank direction

Stop if:

```text
all bank advantages disappear under paired active-capacity controls
or the advantage is explained entirely by the final adaptive readout
or the bank is unused under causal deletion
```

---

# 9. HiPerGator execution

Create:

```text
configs/phase5/stage2_component.yaml
configs/phase5/stage2_capacity.yaml
configs/phase5/stage2_factorial.yaml
configs/phase5/stage2_symbolic.yaml

kam/phase5/stage2_manifest.py
kam/phase5/stage2_run.py
kam/phase5/stage2_aggregate.py
kam/phase5/stage2_stats.py
kam/phase5/stage2_gate.py

scripts/submit_phase5_stage2_hpg.sh
scripts/run_phase5_stage2_local.sh
```

Requirements:

- immutable JSONL or Parquet manifests;
- one independent row per array task;
- array throttling;
- resume completed rows;
- retry only infrastructure failures;
- scientific gate dependencies between stages;
- no shared multi-node SQLite writer;
- record Git commit, dirty state, hardware, environment, wall time, VRAM, and precision.

Before submitting full arrays, profile three jobs per task and scale and write:

```text
reports/phase5/PHASE5_STAGE2_RESOURCE_FORECAST.md
```

Suggested compute ceilings:

```text
component ablation:   250–500 GPU-hours
capacity crossover:   250–600 GPU-hours
factorial search:     350–800 GPU-hours
symbolic replication: 100–300 GPU-hours
```

These are ceilings, not quotas.

---

# 10. Required deliverables

Reports:

```text
reports/phase5/PHASE5_STAGE2_VALIDITY.md
reports/phase5/PHASE5_STAGE2_COMPONENT_REPORT.md
reports/phase5/PHASE5_STAGE2_CAPACITY_REPORT.md
reports/phase5/PHASE5_STAGE2_FACTORIAL_REPORT.md
reports/phase5/PHASE5_STAGE2_SYMBOLIC_REPORT.md
reports/phase5/PHASE5_STAGE2_DECISION_MEMO.md
reports/phase5/PHASE5_STAGE2_REPRODUCIBILITY.md
```

Machine-readable outputs:

```text
results/phase5/stage2/run_manifest.parquet
results/phase5/stage2/all_metrics.parquet
results/phase5/stage2/paired_seed_metrics.parquet
results/phase5/stage2/component_effects.parquet
results/phase5/stage2/capacity_curves.parquet
results/phase5/stage2/factorial_effects.parquet
results/phase5/stage2/support_diagnostics.parquet
results/phase5/stage2/deletion_curves.parquet
```

The final memo must select exactly one:

```text
PROMOTE_LEARNED_SUPPORT_MEMORY
PROMOTE_FIXED_ANCHOR_LEARNED_GEOMETRY
PROMOTE_DATA_CENTERED_KERNEL_BASIS
PROMOTE_ADAPTIVE_READOUT_ONLY
RETAIN_AS_DIAGNOSTIC_ONLY
STOP_KAM_SPECIFIC_DIRECTION
```
