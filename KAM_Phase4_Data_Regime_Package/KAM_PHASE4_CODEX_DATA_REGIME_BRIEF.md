# KAM Phase IV — Data-Regime Mechanism Study

## Codex execution objective

Determine **why persistent-memory and memory-freezing modes behave differently across data families**, rather than running another undirected architecture search.

The current evidence is development-stage only. It suggests strong task–scale–training interactions:

- `prototype_switch`: persistent memory and staged freezing improve at medium scale but can be strongly harmful at small scale.
- `switching_mackey_glass`: persistent memory can hurt static validation while improving post-shift readout adaptation; staged freezing is mildly useful at small scale and harmful at medium scale.
- `switching_narma`: persistent memory is modestly useful at medium scale, and staged freezing is directionally useful at both tested scales.

The next campaign must test whether these differences are explained by:

1. discrete versus continuous predictive regimes;
2. exogenous randomness versus autonomous chaos;
3. observation noise versus process noise;
4. full versus partial observability;
5. finite versus long/chaotic memory;
6. support capacity relative to intrinsic task complexity;
7. fixed-time freezing versus convergence-aware freezing;
8. coordinate mismatch caused by freezing only keys and values while the memory query space continues moving.

Do not describe any hypothesis below as established before the locked confirmatory stage.

---

# 1. Working interpretation of the existing results

## 1.1 Mode definitions

Use these names consistently:

- `D0`: context model with no persistent memory.
- `DD-J`: dot-product context plus dot-product persistent memory, trained jointly through the final step. This is the existing `DD-b`.
- `DD-KV75`: keys and values frozen after 75% of training. This is the existing `DD-b-staged`.
- `DD-P75`: the entire persistent-memory coordinate path freezes after 75%: memory query projection, key/value bank, score parameters, memory output projection, and any memory-specific normalization.
- `DD-K75`: freeze keys only.
- `DD-V75`: freeze values only.
- `DD-SLOW`: joint training with a lower memory learning rate.
- `DD-DRIFT`: freeze the full memory path when a convergence/drift criterion is met.
- `RF-b`: fixed random bank with the same route/value feature dimension.
- `ORACLE-b`: task-specific oracle supports available only for synthetic mechanism tests.

The suffix `-b` means both retrieved values and route features are available to the readout.

## 1.2 Current descriptive pattern

| Data family | Persistent memory: S | Persistent memory: M | Staged freeze: S | Staged freeze: M | Initial interpretation |
|---|---:|---:|---:|---:|---|
| Prototype switch | −14.1% validation | +10.6% validation | −75.1% test | +12.4% test | A discrete bank may help once capacity is sufficient; fixed-time freezing can lock an underfit small bank. |
| Switching Mackey–Glass | −0.3% validation | −28.9% validation | +3.3% test | −9.7% test | The bank may be a local basis on a continuous attractor, not a regime dictionary; larger banks can overpartition or require continued coordinate co-adaptation. |
| Switching NARMA | −1.8% validation | +4.0% validation | +8.9% test | +8.6% test | A finite nonlinear-memory task may benefit from reusable features and from preventing the bank from chasing particular random input realizations. |

Positive percentages mean lower loss for the candidate. These cells have too few pairs for mechanism claims.

## 1.3 Primary scientific hypotheses

### H1 — Regime discreteness

Persistent memory is most useful when the conditional predictor decomposes into recurring, well-separated regimes:

\[
p(y_{t+1}\mid x_{\le t})
\approx
p(y_{t+1}\mid x_{\le t}, r_t),
\qquad
r_t\in\{1,\ldots,R\}.
\]

Expected result: memory benefit grows with regime separation, recurrence frequency, and dwell time.

### H2 — Continuous-manifold penalty

For autonomous chaotic dynamics, the useful predictive structure may vary continuously across an attractor rather than forming a small number of discrete prototypes. A finite bank then behaves as a basis expansion or vector quantizer. Too many supports may overpartition the training attractor and hurt static generalization.

Expected result: persistent-memory benefit decreases as the latent predictive map changes continuously rather than by discrete regime.

### H3 — Exogenous randomness and freeze regularization

Standard NARMA is random-input driven but deterministic conditional on the relevant input/output history. Continued bank movement may fit idiosyncrasies of sampled input histories. Freezing can reduce variance after useful nonlinear-lag features have formed.

Expected result: freezing helps more under exogenous-input randomness or observation noise than under deterministic, continuously evolving dynamics.

### H4 — Process noise is different from observation noise

Observation noise corrupts measurements while preserving the hidden transition law; process noise changes the state itself. A frozen bank may suppress measurement noise but become stale under true process changes.

Expected result: freeze benefit is positive under observation noise and weaker or negative under process noise.

### H5 — Fixed 75% freezing is not task invariant

A fixed fraction of training does not indicate that the bank has converged. Medium models may require more samples than small models, while simple discrete tasks may converge earlier.

Expected result: drift- or validation-triggered freezing is more consistent than a universal 75% boundary.

### H6 — Coordinate mismatch

`DD-KV75` freezes keys and values while the query projection and other memory-path parameters continue changing. The fixed keys can therefore be expressed in a moving coordinate system.

Expected result: `DD-P75` outperforms `DD-KV75` when post-freeze coordinate drift is the cause of degradation.

### H7 — Adaptation utility differs from static utility

Mackey–Glass results suggest the bank may expose features that are not optimal for static prediction but are useful for a fast NLMS readout after a regime transition.

Expected result: memory improves feature reweightability, measured by post-shift adaptation and feature conditioning, even when frozen validation loss is unchanged or worse.

### H8 — Scale acts through support adequacy and optimization

The sign reversal on `prototype_switch` is more likely to depend on support coverage, support-to-regime ratio, training samples per parameter, and convergence speed than on nominal parameter count alone.

Expected result: effects align more strongly with effective supports per predictive regime and sample budget than with `S`/`M` labels.

---

# 2. Mandatory implementation work

## Objective I1 — Reproduce and audit Phase III

1. Recompute all reported tables from raw run artifacts.
2. Verify sign conventions for relative improvement.
3. Confirm exact train/validation/test separation.
4. Confirm that held-out test results never affect checkpoint selection.
5. Export one row per run with:
   - task;
   - data-factor settings;
   - model mode;
   - scale;
   - seed;
   - sample budget;
   - parameter count;
   - support count;
   - context length;
   - optimizer settings;
   - freeze event and reason;
   - static and adaptive metrics.
6. Produce `PHASE3_REPRODUCTION_AUDIT.md`.

Acceptance: every published aggregate can be reconstructed from a checked machine-readable table.

## Objective I2 — Instrument representation and memory dynamics

At every evaluation checkpoint, record:

- key and value drift from initialization;
- interval key/value drift;
- memory-query projection drift;
- memory output projection drift;
- route entropy;
- effective support count;
- dead-support fraction;
- duplicate-support fraction;
- attention mass per support;
- per-support target variance;
- per-support local-linear residual;
- gradients for key, value, query, score, and output parameters;
- gradient mean, variance, and signal-to-noise ratio by parameter group;
- condition number and effective rank of route features;
- condition number and effective rank of the final adaptive feature matrix;
- function drift on a fixed anchor set.

Define anchor-set function drift as

\[
\Delta_f(t_1,t_2)=
\frac{
\|\Psi_{t_2}(X_A)-\Psi_{t_1}(X_A)\|_F
}{
\|\Psi_{t_1}(X_A)\|_F+\epsilon
}.
\]

Also decompose function drift by intervention:

1. update query path only;
2. update keys only;
3. update values only;
4. update the entire memory path.

Acceptance: a run can identify whether performance changes coincide with useful support movement, noisy gradients, or coordinate mismatch.

## Objective I3 — Implement freeze policies

Implement and unit-test:

- joint training;
- key-only freeze;
- value-only freeze;
- key/value freeze;
- full-memory-path freeze;
- memory learning-rate ratios \(\{1, 0.3, 0.1, 0.03\}\);
- drift-triggered freeze;
- plateau-triggered freeze;
- periodic freeze/unfreeze;
- exponential-moving-average keys and values.

A drift-triggered freeze may fire only when all hold for at least \(K\) evaluations:

\[
\Delta_{\text{keys}} < \delta_k,
\quad
\Delta_{\text{values}} < \delta_v,
\quad
\mathrm{SNR}_{\nabla,\mathrm{memory}} < \delta_g,
\]

and validation loss has not improved by more than \(\delta_L\).

All thresholds are development hyperparameters. The frozen event must be logged.

## Objective I4 — Add task-factor controls

Implement generators exposing the factors below independently.

### Controlled prototype system

Vary:

- regimes: \(R\in\{2,4,8,16\}\);
- regime separation;
- abrupt versus continuous interpolation;
- recurrence probability;
- dwell time;
- overlap between regime state distributions;
- observation noise;
- process noise;
- full versus partial regime observability;
- support-to-regime ratio.

Include oracle supports representing the true regime predictors.

### Controlled symbolic regime language

Create a token generator with:

- deterministic to high-entropy transitions;
- hidden recurring regimes;
- controllable emission overlap;
- known Bayes entropy;
- optional explicit regime token;
- abrupt and gradual regime switches.

This directly tests the user's stochastic-language hypothesis without introducing natural-language confounds.

### Switching NARMA family

Vary:

- order \(p\in\{5,10,20,30\}\);
- context length \(L\in\{p/2,p,2p,4p\}\);
- input visible versus input hidden from the model;
- input distribution;
- coefficient regimes;
- observation noise;
- process noise;
- abrupt versus gradual switching;
- recurrence of prior regimes.

### Switching Mackey–Glass family

Vary:

- delay \(\tau\);
- context-to-delay ratio \(L/\tau\);
- feedback parameters;
- periodic versus chaotic settings;
- observation noise;
- process noise;
- full delay-vector access versus compressed/aliased observations;
- abrupt and gradual parameter switches;
- recurring and novel regimes.

### Optional continuous-chaos replication

Add Lorenz-63 or Rössler with state-preserving parameter switches. Use this only as a replication family after the generators above pass numerical tests.

---

# 3. Experiment stages

## Stage A — Existing-checkpoint mechanism audit

Use existing Phase III checkpoints where possible.

Required analyses:

1. correlate memory drift with held-out error;
2. correlate memory gradient SNR with the benefit of freezing;
3. measure route-feature conditioning before and after shift;
4. test whether NLMS gains are explained by improved feature conditioning;
5. compare support alignment with:
   - true discrete regimes;
   - phase-space clusters;
   - local predictive Jacobians;
6. replay the final 25% of training under counterfactual freezes:
   - KV frozen;
   - full path frozen;
   - keys frozen;
   - values frozen.

Deliverables:

- `PHASE3_MECHANISM_AUDIT.md`
- `memory_drift_effects.parquet`
- `gradient_snr.parquet`
- `feature_conditioning.parquet`
- `counterfactual_freeze_results.parquet`

## Stage B — Factorial mechanism screen

Purpose: identify which data properties predict memory and freeze utility.

Primary models:

```text
D0
DD-J
DD-KV75
DD-P75
DD-DRIFT
RF-b
ORACLE-b  # controlled synthetic tasks only
```

Scales:

```text
S: approximately 250k parameters
M: approximately 1M parameters
```

Development seeds:

```text
3 seeds per cell
```

Fidelity:

```text
25%, 50%, 100% sample budgets
```

Target size:

- 500–900 completed or pruned runs;
- balanced trial counts across models;
- identical generated streams for paired comparisons.

Do not cross every factor naïvely. Use a fractional factorial or Latin-hypercube design that preserves estimation of:

- model × regime-discreteness;
- model × noise-type;
- model × observability;
- model × memory-horizon;
- freeze-policy × gradient-SNR;
- scale × support-to-regime ratio.

Fit an explanatory mixed-effects model after the screen:

\[
\text{relative improvement}
\sim
\text{regime discreteness}
+
\text{noise type}
+
\text{observability}
+
\text{memory horizon}
+
\text{scale}
+
\text{freeze policy}
+
\text{registered interactions}
+
(1|\text{seed}).
\]

Use this model descriptively for mechanism discovery, not as confirmatory proof.

## Stage C — Freeze-boundary and coordinate-consistency search

For each data family, promote the two most informative conditions from Stage B.

Compare:

```text
DD-J
DD-K25
DD-K50
DD-K75
DD-V25
DD-V50
DD-V75
DD-KV25
DD-KV50
DD-KV75
DD-P25
DD-P50
DD-P75
DD-P90
DD-SLOW
DD-DRIFT
```

Required controls:

- same architecture and parameter count;
- same training streams;
- same sample budget;
- no checkpoint selection on held-out test;
- training-sample-matched and wall-clock-matched views.

Target size:

- 250–450 runs after pruning;
- 3 seeds for search;
- 5 seeds for the top three policies per family.

Primary outcome: held-out test NMSE.

Mechanism outcomes:

- post-freeze coordinate drift;
- anchor-set function jump;
- support gradient SNR;
- route entropy;
- support utilization;
- adaptive late post-transition NMSE.

## Stage D — Long-training and scale study

Purpose: determine whether current sign changes are undertraining effects.

Variants:

```text
D0
DD-J
best freeze policy from Stage C
RF-b
```

Scales:

```text
S: 250k
M: 1M
L: 4M
optional XL: 16M
```

Sample budgets:

```text
S: 5M, 20M observations
M: 10M, 40M observations
L: 20M, 80M observations
XL: 100M observations if justified
```

Run:

- 5 development seeds;
- at least 100 logged evaluation points;
- no early stopping before 40% of budget;
- both sample-matched and compute-matched comparisons.

Fit scaling curves against:

- observations;
- optimizer steps;
- GPU-hours;
- parameters;
- effective support count;
- supports per regime or per attractor cluster.

## Stage E — Locked confirmatory tests

Lock no more than two primary comparisons based on Stages A–D.

Recommended structure:

1. one discrete/recurring regime condition where persistent memory is predicted to help;
2. one continuous-chaotic condition where persistent memory is predicted not to help or to require joint adaptation.

Use:

- 10–16 new training seeds;
- at least 10 held-out streams or schedules per seed;
- paired inference;
- one primary endpoint per condition;
- paired bootstrap confidence intervals;
- exact paired permutation tests;
- equivalence test with a ±5% margin.

The training seed is the independent inference unit. Aggregate schedules within seed.

---

# 4. Core metrics and analyses

## Static performance

- MSE, NMSE, NRMSE, MAE;
- bias;
- p90/p95 absolute error;
- \(R^2\);
- correlation;
- held-out test loss;
- stationary degradation relative to `D0`.

## Online adaptation

Every transition must follow:

```text
predict -> score -> reveal target -> update
```

Report:

- early post-transition loss;
- cumulative post-transition loss;
- late post-transition loss;
- recovery time;
- reacquisition time when a prior regime returns;
- adaptation update FLOPs;
- adaptation memory;
- NLMS, SGD, and RLS under identical features.

## Memory semantics

Report both discrete and continuous alignment:

- adjusted mutual information with regime labels;
- support purity with a permutation null;
- mutual information with attractor/phase clusters;
- local predictive Jacobian similarity;
- within-support target variance;
- within-support local-linear residual;
- causal top-versus-random deletion;
- support activation fingerprints across seeds.

## Optimization mechanism

- gradient SNR by parameter group;
- key/value/query drift;
- anchor-set feature drift;
- train–validation–test gap;
- route-feature effective rank;
- feature Gram condition number;
- NLMS convergence prediction from feature covariance.

## Efficiency

- trainable parameters;
- forward FLOPs;
- training samples/s;
- batch-1 latency;
- throughput latency;
- peak VRAM;
- GPU-hours;
- disk usage;
- quality versus wall time.

---

# 5. Decision rules

## Support the discrete-regime hypothesis only if

- memory benefit increases with controlled regime separation or recurrence;
- learned memory beats fixed random features;
- supports are causally used;
- support assignments align with regimes beyond a null;
- results replicate on the symbolic regime language or prototype system.

## Support the stochastic-freeze hypothesis only if

- freeze benefit increases with observation noise or exogenous-input randomness;
- the same effect does not appear equally under deterministic matched controls;
- late memory-gradient SNR decreases as freeze benefit increases;
- full-memory-path freezing or slow memory learning reproduces the gain.

## Support the coordinate-mismatch hypothesis only if

- `DD-P75` consistently outperforms `DD-KV75`;
- post-freeze query/key coordinate drift predicts `DD-KV75` degradation;
- freezing the full path reduces anchor-set function drift.

## Support the adaptation-feature hypothesis only if

- persistent memory improves matched NLMS/RLS adaptation;
- static loss may remain neutral or worse;
- route-feature covariance is better conditioned or more linearly separable after shift;
- learned routes beat equal-dimensional random routes.

## Stop the KAM-specific direction if

- learned memory does not beat random features;
- causal deletion fails;
- task-factor effects are inconsistent after controlled tests;
- persistent memory is equivalent within ±5% but has material overhead;
- any observed gain is fully explained by a generic adaptive readout.

---

# 6. Compute and scheduling plan

## Local PC lane

Use for:

- unit tests;
- generator verification;
- Stage A audits;
- small-scale pilot runs;
- figure generation;
- resuming failed jobs;
- overnight fractional-factor screen with a wall-clock limit.

The local runner must:

- checkpoint periodically;
- stop launching new trials near the deadline;
- record GPU and environment metadata;
- resume from a static manifest;
- never overwrite completed runs.

## HiPerGator lane

Use job arrays for Stages B–E.

Requirements:

- static manifest with one independent row per job;
- output directories keyed by immutable run ID;
- array throttling;
- SLURM dependency gates between stages;
- no shared multi-node SQLite writer;
- aggregation only after job arrays complete;
- retries only for infrastructure failures, never silent metric failures.

Before submission, profile three pilot jobs per scale and forecast:

- wall time;
- VRAM;
- storage;
- expected completed/pruned fraction;
- total GPU-hours.

Suggested ceilings:

```text
Stage A audit:                  10–30 GPU-hours
Stage B factorial screen:     300–700 GPU-hours
Stage C freeze-policy search: 200–500 GPU-hours
Stage D long-scale study:     400–1000 GPU-hours
Stage E confirmation:         250–700 GPU-hours
```

These are ceilings, not quotas. Reduce the queue when scientific resolution is achievable more cheaply.

---

# 7. Required repository deliverables

Codex must create or update:

```text
configs/phase4/
manifests/phase4/
scripts/run_phase4_local.sh
scripts/submit_phase4_hpg.sh
scripts/aggregate_phase4.py
scripts/gate_phase4.py
src/.../freeze_policies.py
src/.../controlled_generators.py
src/.../memory_diagnostics.py
tests/test_freeze_policies.py
tests/test_controlled_generators.py
tests/test_memory_diagnostics.py
```

Required reports:

```text
PHASE3_REPRODUCTION_AUDIT.md
PHASE3_MECHANISM_AUDIT.md
PHASE4_FACTORIAL_REPORT.md
PHASE4_FREEZE_POLICY_REPORT.md
PHASE4_SCALING_REPORT.md
PHASE4_CONFIRMATORY_REPORT.md
PHASE4_DECISION_MEMO.md
PHASE4_REPRODUCIBILITY.md
```

Required machine-readable outputs:

```text
run_manifest.parquet
all_metrics.parquet
memory_drift_effects.parquet
gradient_snr.parquet
feature_conditioning.parquet
support_diagnostics.parquet
causal_deletion.parquet
factorial_effects.parquet
freeze_policy_effects.parquet
scaling_curves.parquet
confirmatory_seed_metrics.parquet
```

## Final memo questions

The final decision memo must answer:

1. Which data properties predict persistent-memory utility?
2. Is stochasticity itself causal, or is the effect better explained by noise type, observability, or regime geometry?
3. Why does scale reverse the result on the prototype task?
4. Does freezing help because it regularizes noise, because the bank has converged, or because it prevents destructive basis drift?
5. Does freezing only keys and values create coordinate mismatch?
6. Is the bank acting as a regime dictionary, a local basis, or merely extra random-like features?
7. Is its main value static prediction or online adaptation?
8. Which architecture and training policy, if any, deserves further work?
9. Which negative conclusion is justified if no mode passes?

The final decision must be one of:

```text
PROMOTE_DISCRETE_REGIME_MEMORY
PROMOTE_TASK_SPECIFIC_ADAPTIVE_FEATURES
RETAIN_AS_DIAGNOSTIC_ONLY
SIMPLIFY_TO_GENERIC_ADAPTIVE_READOUT
STOP_KAM_SPECIFIC_DIRECTION
```
