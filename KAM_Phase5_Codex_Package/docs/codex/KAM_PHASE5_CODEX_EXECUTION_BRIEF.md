# KAM Phase V — Codex Execution Brief

## Mission

Implement and deploy the next KAM experiment campaign in `Jibby2k1/KAM`.

The current evidence shows that a persistent nonlinear bank can help some recurring dynamical tasks, but a frozen random bank often matches the learned bank. Phase V must determine whether the gain comes from:

1. learned supports;
2. a fixed nonlinear feature expansion;
3. extra readout dimensions;
4. recurrence and reuse;
5. online adaptation;
6. or an invalid comparison caused by current capacity/data controls.

Do not launch the expensive campaign until the validity fixes and unit tests below pass.

---

# 1. Required repository changes

## 1.1 Active capacity matching

Create:

```text
kam/capacity.py
tests/test_capacity.py
```

Modify:

```text
kam/model.py
kam/factory.py
kam/run_suite.py
```

Requirements:

- Stop using unused `capacity_padding` for scientific comparisons.
- Retain legacy loading compatibility, but report:
  - `active_parameter_count`;
  - `padding_parameter_count`;
  - `total_parameter_count`;
  - approximate forward FLOPs.
- Implement active matching by searching discrete combinations of:
  - `d_model`;
  - `ffn_expansion`;
  - `num_layers`;
  - `route_projection_dim`;
  - optional readout hidden width.
- Match active trainable parameters within 1% where feasible.
- Primary tables must reject rows with nonzero padding parameters.
- Provide same-width and active-parameter-matched views.

Acceptance:

```bash
pytest -q tests/test_capacity.py
```

must verify that matched models use their counted parameters in the forward graph.

## 1.2 Fixed route dimension

Modify `KAMBlock` and `KAMSequenceModel` so primary experiments use:

```yaml
route_features: projected
route_projection_dim: 64
```

or another fixed dimension selected before the comparison.

Requirements:

- The same final readout dimension across D0 and all bank variants.
- Add a matched learned projection for D0 when necessary.
- Keep raw routes for diagnostics only.

## 1.3 Orthogonal controlled streams

Create:

```text
kam/data/controlled_regimes.py
tests/test_controlled_regimes.py
```

The generator schema must independently control:

```text
regime_count
regime_separation
return_probability
dwell_length
transition_type: abrupt | gradual
observation_noise
process_noise
input_noise
observability: full | partial | hidden_driver
context_to_true_memory_ratio
```

Generate **independent** train, validation, test, and prequential streams.

Each stream must save:

- exact schedule;
- regime boundaries;
- generator parameters;
- stream seed;
- Bayes/noise metadata where available.

Unit tests must verify that changing one factor does not silently change another.

## 1.4 Ordered and shuffled protocols

Support two explicitly different training modes:

```text
iid_window_training
ordered_stream_training
```

- `iid_window_training` may shuffle windows and tests representation quality.
- `ordered_stream_training` preserves chronology and tests acquisition, forgetting, recurrence, and reacquisition.
- Never call shuffled-window training an online-memory test.
- Ordered evaluation must use:
  `predict → score → reveal target → update`.

## 1.5 Correct evaluation semantics

Modify `kam/run_suite.py` and aggregation:

- global MSE/NMSE/NRMSE from all collected predictions;
- reload the validation-selected checkpoint before held-out test;
- save final-checkpoint and best-checkpoint results separately;
- report regime-stratified metrics;
- aggregate held-out schedules within training seed;
- use training seed as the inferential unit.

---

# 2. New model/control variants

Add compact labels to `kam/factory.py`.

| Label | Keys | Values | Memory coordinate path | Purpose |
|---|---|---|---|---|
| `D0` | none | none | none | no-bank baseline |
| `DD-L` | learned | learned | learned | fully learned bank |
| `RF-KV` | fixed random | fixed random | learned | current `RF-b`, renamed clearly |
| `RF-FULL` | fixed random | fixed random | fixed | genuine fixed random feature map |
| `RK-LV` | fixed random | learned | learned | value-learning contribution |
| `LK-RV` | learned | fixed random | learned | key-learning contribution |
| `KC-LV` | fixed data centers | learned | learned/fixed as ablation | data-initialized kernel basis |
| `RFF` | random Fourier features | n/a | fixed | non-attention random-feature baseline |
| `DD-PF` | learned then full-path frozen | learned then frozen | fully frozen after boundary | coordinate-consistent freeze |
| `DD-DRIFT` | learned | learned | drift-triggered freeze | convergence-aware freeze |

For all feature-bank variants:

- support/feature count is controlled independently;
- route output dimension is fixed;
- readout architecture is matched;
- active capacity is matched;
- feature initialization norms are matched where meaningful.

Implement fixed-center initialization options:

```text
random_normal
sampled_training_points
kmeans
farthest_point
```

---

# 3. Core experiment matrix

## 3.1 Tasks

### A. Controlled prototype regimes

Vary independently:

```text
regimes: 2, 4, 8, 16
return_probability: 0, 0.25, 0.5, 0.75, 1
separation: low, medium, high
dwell: short, medium, long
observation_noise: 0, low, high
process_noise: 0, low
observability: full, partial
```

Include `ORACLE` centers based on the true synthetic regimes.

### B. Switching Mackey–Glass

Vary:

```text
tau
beta
periodic_vs_chaotic
return_probability
context_length / maximum_delay
observation_noise
process_noise
full_delay_window_vs_aliased_window
```

### C. Switching NARMA

Vary:

```text
order: 5, 10, 20, 30
context_length / order
driver_visible: true, false
coefficient separation
input distribution
observation noise
process noise
return probability
```

### D. Controlled symbolic regime language

Create a small causal language with:

```text
known hidden regimes
controlled transition entropy
controlled emission overlap
return probability
explicit-regime-token ablation
```

This tests stochasticity without natural-language confounds.

## 3.2 Primary model shortlist

Use for broad search:

```text
D0
DD-L
RF-KV
RF-FULL
KC-LV
RFF
```

Use only after shortlist:

```text
RK-LV
LK-RV
DD-PF
DD-DRIFT
```

## 3.3 Scale axes

Do not bundle scale variables.

Sweep independently:

```text
active parameters: 250k, 1M, 4M
supports/features: 16, 32, 64, 128, 256
route dimension: fixed at 64 for primary comparisons
context ratio: 0.5x, 1x, 2x, 4x true memory
training samples per active parameter
```

The 16M scale is conditional on a positive 4M result.

---

# 4. Execution stages

## Stage 0 — Validity gate

Deliver:

```text
reports/phase5/PHASE5_VALIDITY_AUDIT.md
results/phase5/validity_checks.json
```

Gate requirements:

- active capacity matching passes;
- zero padding in primary rows;
- best-checkpoint test evaluation verified;
- orthogonal generator tests pass;
- ordered protocol preserves sequence;
- global NMSE verified against direct NumPy calculation;
- fixed route dimension verified;
- all variants smoke-test on CPU and GPU.

Do not submit the full HPG array before this gate passes.

## Stage 1 — Mechanism pilot

Run:

```text
4 tasks × 6 variants × 2 scales × 3 seeds
```

with a small fractional-factor set.

Target:

```text
150–250 completed runs
```

Purpose:

- verify signal direction;
- measure variance;
- profile runtime/VRAM;
- identify unusable controls.

## Stage 2 — Multi-fidelity factorial search

Use Optuna or a static low-discrepancy design with reproducible manifests.

Target:

```text
400–800 completed/pruned trials
```

Fidelities:

```text
20%, 50%, 100% sample budget
```

Search:

- feature/support count;
- width/depth allocation;
- memory learning rate;
- route projection;
- bandwidth/temperature;
- center initialization;
- freeze policy;
- context ratio.

Use identical streams for paired variants.

## Stage 3 — Long-training scale study

Promote at most three bank mechanisms plus D0.

Run:

```text
250k, 1M, 4M active parameters
5 seeds
at least 10 independent held-out streams per seed
```

Suggested observation budgets:

```text
250k: 5M and 20M
1M:   10M and 40M
4M:   20M and 80M
```

Report sample-matched and GPU-hour-matched curves.

## Stage 4 — Ordered recurrence/adaptation study

For promoted variants, use:

```text
A → B → A → C → A
```

and gradual-switch controls.

Every adapter must use the same frozen backbone features:

```text
frozen
NLMS
SGD
RLS
```

Primary metrics:

- late post-transition NMSE;
- cumulative excess loss;
- reacquisition time for returning regimes;
- stationary degradation;
- adapter FLOPs and state.

## Stage 5 — Locked confirmation

Lock at most two claims:

1. learned bank versus best fixed-feature control;
2. best fixed-feature bank versus D0.

Use:

```text
10–16 new training seeds
10 held-out streams/schedules per seed
one primary endpoint per claim
paired bootstrap CI
exact paired permutation test
±5% equivalence test
```

---

# 5. Diagnostics

Required per checkpoint:

- route entropy and effective feature count;
- dead/duplicate support fraction;
- feature Gram effective rank and condition number;
- support activation fingerprints;
- cross-seed Hungarian matching;
- top/random/bottom deletion;
- learned-versus-random feature deletion;
- support nearest contexts;
- regime mutual information with permutation null;
- within-support target variance;
- local-linear residual;
- feature drift on a fixed anchor set;
- gradient SNR for keys, values, query path, and readout.

For fixed features, use the same diagnostics where meaningful.

The decisive semantic test is not whether supports look interpretable. It is whether learned supports beat fixed features in accuracy, adaptation, compression, sample efficiency, stability, or causal faithfulness.

---

# 6. Statistical decision rules

## Promote learned memory only if

- `DD-L` beats the strongest fixed-feature control by at least 5%;
- paired 95% CI excludes zero;
- learned features achieve the same performance with materially fewer supports, or show a clear adaptation/sample-efficiency advantage;
- causal deletion and support use are nondegenerate;
- overhead is reported and justified.

## Promote stable fixed features if

- `RF-FULL`, `KC-LV`, or `RFF` beats D0;
- learned supports are equivalent within ±5%;
- fixed features are cheaper, more stable, or easier to adapt.

## Simplify to generic adaptive readout if

- all bank advantages disappear under active-capacity and route-dimension controls;
- NLMS/RLS explains the remaining gains for every backbone.

## Stop the KAM-specific direction if

- neither learned nor fixed banks beat D0 at matched active capacity/compute;
- deletion shows the bank is unused;
- gains do not replicate across controlled streams.

---

# 7. Required code and report paths

Create:

```text
kam/phase5/
  manifest.py
  run_array.py
  aggregate.py
  gate.py
  stats.py
kam/data/controlled_regimes.py
kam/capacity.py
configs/phase5/
scripts/submit_phase5_hpg.sh
scripts/run_phase5_local.sh
tests/test_phase5.py
tests/test_capacity.py
tests/test_controlled_regimes.py
reports/phase5/
results/phase5/
```

Final reports:

```text
PHASE5_VALIDITY_AUDIT.md
PHASE5_MECHANISM_REPORT.md
PHASE5_FACTORIAL_REPORT.md
PHASE5_SCALING_REPORT.md
PHASE5_ADAPTATION_REPORT.md
PHASE5_CONFIRMATORY_REPORT.md
PHASE5_DECISION_MEMO.md
PHASE5_REPRODUCIBILITY.md
```

Machine-readable outputs:

```text
run_manifest.parquet
all_metrics.parquet
paired_seed_metrics.parquet
feature_diagnostics.parquet
deletion_curves.parquet
adaptation_metrics.parquet
scaling_curves.parquet
confirmatory_metrics.parquet
```

---

# 8. HiPerGator deployment requirements

- Static JSONL/Parquet manifest; one row per independent job.
- SLURM arrays with concurrency throttling.
- Immutable run IDs.
- Resume completed rows without rerunning.
- Separate validity, pilot, search, scaling, adaptation, confirmation, aggregation, and reporting jobs.
- Use job dependencies so failed scientific gates block downstream arrays.
- Profile three jobs per scale before full submission.
- Record allocation, partition, GPU, precision, environment, commit, dirty state, wall time, and storage.

Suggested upper ceilings:

```text
validity + pilots:       20–50 GPU-hours
factorial search:       300–700 GPU-hours
long scale study:       400–900 GPU-hours
ordered adaptation:     200–500 GPU-hours
confirmation:           250–600 GPU-hours
```

These are ceilings, not spending targets.

---

# 9. Final required conclusion

The final memo must select exactly one:

```text
PROMOTE_LEARNED_SUPPORT_MEMORY
PROMOTE_FIXED_KERNEL_FEATURE_BANK
PROMOTE_GENERIC_RANDOM_FEATURE_BANK
PROMOTE_ADAPTIVE_READOUT_ONLY
RETAIN_AS_DIAGNOSTIC_ONLY
STOP_KAM_SPECIFIC_DIRECTION
```
