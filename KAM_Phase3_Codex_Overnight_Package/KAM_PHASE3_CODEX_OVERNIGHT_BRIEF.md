# Kernel Adaptive Memory Phase III
## Codex Overnight Execution Brief

**Status:** Development continuation with one narrow promotable hypothesis  
**Date:** 2026-07-24  
**Primary question:** Does a learned persistent support bank improve online adaptation in recurring nonlinear dynamical systems after exact capacity, compute, adapter, causal-use, and statistical controls?  
**Primary candidate:** `DD-b + NLMS`  
**Primary baseline:** `D0 + NLMS`  
**Radial ablation:** `DR-b + NLMS`

---

## 1. Why this phase exists

Phase II completed stationary and switching screens, exact parameter matching, held-out prequential evaluation, mechanism-language tests, timing studies, and substantial hyperparameter search. It did **not** establish a corrected advantage for radial context attention or radial persistent memory.

One signal remains worth a final, disciplined investigation:

> Dot-product context attention plus a learned persistent memory bank may expose reusable routing features that improve inexpensive online adaptation when nonlinear regimes recur.

The Phase II held-out Mackey--Glass screen favored `DD-b + NLMS` over `D0 + NLMS` in mean late loss, while `DR-b` did not improve on `DD-b`. The effect did not pass the registered corrected gate. Therefore Phase III must not resume a broad search for a generally superior “KAM transformer.” It must determine whether the narrower persistent-memory mechanism is real, causal, stable, scalable, and worth its overhead.

This brief gives Codex enough work to build, validate, and queue an overnight local-GPU and HiPerGator campaign. Large exploratory runs are permitted, but exploratory results must never be reported as confirmatory evidence.

---

## 2. Scientific claims and falsifiers

### H1 — Persistent-memory adaptation

At exact parameter matching and with the same NLMS adapter,

\[
\mathrm{DD\mbox{-}b} + \mathrm{NLMS}
\]

has lower late post-transition normalized loss than

\[
\mathrm{D0} + \mathrm{NLMS}
\]

on recurring Mackey--Glass schedules, without more than 5% stationary degradation.

**Primary practical threshold:** at least 15% lower late post-transition NMSE.

### H2 — Radial-memory geometry

At otherwise matched architecture and training,

\[
\mathrm{DR\mbox{-}b} + \mathrm{NLMS}
\]

improves on or is meaningfully different from

\[
\mathrm{DD\mbox{-}b} + \mathrm{NLMS}.
\]

This is secondary. Existing evidence suggests radial memory may be neutral or harmful.

### H3 — Causal and stable support use

The learned support routes are not merely decorative attention weights or generic feature expansion. High-weight supports must have larger causal effects than matched random supports, learned memory features must beat equal-dimensional random features, and support activation functions must show above-null stability across seeds.

### Hard falsifiers

Stop the KAM-specific direction if any of the following is established:

1. `DD-b + NLMS` is practically equivalent to `D0 + NLMS` within a preregistered ±5% relative late-loss margin while carrying material overhead.
2. Top-support deletion is not more damaging than random deletion.
3. Learned support features do not outperform matched fixed-random features.
4. Supports are extensively dead, duplicated, or unstable across seeds.
5. Any apparent advantage disappears under new training seeds and new held-out schedules.
6. The advantage is limited to development Mackey--Glass streams and does not replicate on a second nonlinear recurring system.

---

## 3. Terminology and model registry

Retain the existing compact labels.

| Label | Context score | Persistent memory | Output path |
|---|---|---|---|
| `D0` | dot | none | hidden state |
| `DD-v` | dot | dot | retrieved values only |
| `DD-a` | dot | dot | routes only |
| `DD-b` | dot | dot | values and routes |
| `DR-v` | dot | radial | retrieved values only |
| `DR-a` | dot | radial | routes only |
| `DR-b` | dot | radial | values and routes |
| `RF-b` | dot | frozen random bank | values and routes |

`RF-b` is required in Phase III. It must have the same number and dimension of support features as the learned bank. At least two random controls are required:

- frozen random keys with trainable readout only;
- frozen random keys and values with only the final readout trainable.

The primary architecture set is deliberately small:

```text
D0, DD-b, DR-b, RF-b
```

`DD-v`, `DD-a`, `DR-v`, and `DR-a` are mechanism probes, not primary scale-up candidates.

---

## 4. Non-negotiable experimental rules

1. **Prequential ordering:** every online transition evaluation must perform

   ```text
   predict -> score -> reveal target -> update
   ```

   The target must never influence the prediction being scored.

2. **Independent unit:** the training seed is the inferential unit. Multiple schedules and transitions are nested observations and must be aggregated within training seed or handled through a hierarchical bootstrap.

3. **Development versus confirmation:**
   - Existing Phase II seeds and all hyperparameter-search seeds are development data.
   - Confirmatory training seeds, schedule seeds, and stream seeds must be new and locked before confirmatory execution.

4. **Exact matching:** primary comparisons must be exact or within 0.5% in trainable parameter count. Also report measured training and inference cost; parameter matching alone is not sufficient.

5. **Adapter matching:** every backbone in an adaptation comparison receives the same adapter family and the same tuning budget.

6. **No silent exclusions:** crashes, NaNs, OOMs, pruned runs, and corrupted checkpoints must remain in the manifest with explicit status and traceback.

7. **No claim from a search winner:** hyperparameter search selects candidates; it does not provide inferential evidence.

8. **Atomic outputs:** every job writes to a temporary directory and atomically renames it only after validation. A partially written run must never appear completed.

9. **Resume safety:** all queues must be idempotent. Re-running a manifest skips validated completed rows and retries only failed or missing rows.

10. **No new support-plasticity mechanism yet:** do not add online support replacement, birth/death, or coefficient transport unless the current fixed-bank mechanism passes the causal-use gate.

---

## 5. Repository work packages

Codex must first inspect the repository and preserve current commands and artifacts. Implement the following work packages as independent, testable units.

### WP0 — Repository and evidence audit

Create:

```text
reports/phase3/PHASE2_EVIDENCE_AUDIT.md
results/phase3/input_inventory.parquet
results/phase3/checkpoint_inventory.parquet
```

The audit must:

- locate every Phase II checkpoint, resolved config, metrics file, transition trace, deletion file, and search database;
- hash all source artifacts used in Phase III;
- map checkpoints to task, variant, suffix, seed, parameter count, and training commit;
- identify missing or incompatible checkpoints;
- recompute the headline Phase II paired tables from raw files rather than copying report text;
- verify that the reported 50 stationary runs, 50 switching runs, 90 exact-capacity runs, 1260 held-out transition rows, and mechanism-language runs are internally consistent;
- flag any mismatch before new training begins.

**Acceptance test:** a clean audit command reconstructs the Phase II summary tables from raw artifacts and exits nonzero on missing required inputs.

---

### WP1 — Unified Phase III manifest and runner

Create a manifest-driven interface:

```bash
kam-phase3 build-manifest --config configs/phase3/<suite>.yaml
kam-phase3 run-row --manifest <manifest.parquet> --row-id <integer>
kam-phase3 aggregate --run-root results/phase3/<suite>
kam-phase3 gate --gate-config configs/phase3/gates.yaml
kam-phase3 report --run-root results/phase3
```

Each manifest row must contain at least:

```text
run_id
stage
task
variant
memory_output
parameter_target
actual_parameters
model_scale
context_length
support_count
training_seed
stream_seed
schedule_seed
adapter
adapter_seed
precision
max_steps
sample_budget
walltime_budget_minutes
config_hash
git_commit
output_path
```

Every completed run must save:

```text
resolved_config.yaml
status.json
metrics.json
history.parquet
predictions.parquet or compressed npz
checkpoint_best.pt
checkpoint_final.pt
hardware.json
runtime.json
```

Use a run-level lock and atomic completion marker.

**Required tests:** duplicate invocation, interrupted run, resume from checkpoint, malformed manifest row, deterministic data regeneration, and failed-run retry.

---

### WP2 — Existing-checkpoint causal audit

Run on all compatible `DD-b` and `DR-b` Phase II switching checkpoints and held-out schedules.

#### WP2.1 Support deletion

For each prediction and each head, intervene at:

```text
k = 1, 2, 4, 8, 16, 25% of bank, 50% of bank
```

Evaluate:

- delete top-weighted supports;
- delete bottom-weighted supports;
- delete matched random supports using at least 100 random draws per checkpoint/schedule;
- set selected support logits to `-inf` and renormalize;
- zero only selected support values while preserving routes;
- hide only selected route features from the readout;
- remove both value and route pathways.

Keep the trained readout fixed during each intervention.

Primary faithfulness statistic:

\[
\Delta_{\mathrm{TR}}(k)
= L_{\mathrm{top}}(k)-L_{\mathrm{random}}(k).
\]

Also calculate an integrated top-versus-random deletion area over the registered values of \(k\).

#### WP2.2 Key/value/routing interventions

Evaluate:

- shuffle keys across supports;
- shuffle values across supports;
- shuffle key-value pairs jointly;
- replace routes by their validation mean;
- replace routes by a uniform distribution;
- permute routes independently per sample;
- replace learned keys with matched-norm random keys;
- replace learned values with matched-distribution random values.

#### WP2.3 Adapted versus frozen faithfulness

Run interventions both:

- before online adaptation;
- after NLMS has adapted through each transition segment.

This determines whether the adaptive readout actually uses the support features.

#### WP2.4 Outputs

```text
results/phase3/audit/deletion_curves.parquet
results/phase3/audit/intervention_metrics.parquet
reports/phase3/CAUSAL_SUPPORT_AUDIT.md
```

**Gate A1:** seed-aggregated top deletion must be more damaging than random deletion, with a paired 95% confidence interval above zero, for the combined memory pathway of `DD-b`.

---

### WP3 — Support stability and noncollapse audit

Use a fixed anchor corpus for each task. Generate at least 10,000 anchor contexts from held-out schedule families and preserve their exact seeds.

For every head-support unit \(i\), compute the activation fingerprint

\[
f_i=[a_i(x_1),\ldots,a_i(x_N)].
\]

Match units across seeds with the Hungarian algorithm using fingerprint correlation as the similarity score. Because attention heads can also permute, match over the combined head-support index unless the implementation provides a canonical head ordering.

Report:

- matched fingerprint Pearson and Spearman correlation;
- nearest-context overlap at top 10, 50, and 100 anchors;
- matched value-vector cosine similarity;
- top-1 selection frequency;
- mean route mass;
- per-sample and global effective support count;
- dead-support fraction;
- near-duplicate fraction;
- support Gram-matrix effective rank;
- route entropy distribution;
- stability before and after online adaptation.

Definitions:

- dead support: mean route mass `< 1e-4` **and** top-1 frequency `< 1e-3`;
- near duplicate: activation fingerprint correlation `> 0.995` or normalized key distance `< 0.1` bandwidth units;
- effective count: inverse participation ratio.

Nulls:

- permute anchor ordering independently by seed;
- match against untrained random banks;
- match against `RF-b` banks.

Outputs:

```text
results/phase3/audit/support_fingerprints.zarr
results/phase3/audit/support_stability.parquet
results/phase3/audit/support_nulls.parquet
reports/phase3/SUPPORT_STABILITY_AUDIT.md
```

**Gate A2:** learned-bank stability must exceed the random-bank and permutation nulls, and fewer than 25% of supports may be dead or duplicate at the selected scale.

---

### WP4 — Feature-path and random-feature probes

For each Phase II switching checkpoint, extract:

- hidden state only;
- routes only;
- retrieved values only;
- hidden + routes;
- hidden + values;
- hidden + routes + values;
- matched fixed random features;
- learned routes with shuffled labels or shuffled keys as negative controls.

Fit identical frozen probes with:

- ridge regression;
- NLMS;
- SGD linear regression;
- RLS where dimension permits.

Use nested train/validation streams for regularization and adapter hyperparameters. Do not tune on the held-out evaluation schedules.

Outputs:

```text
results/phase3/audit/frozen_probe_metrics.parquet
results/phase3/audit/random_feature_controls.parquet
reports/phase3/FEATURE_PATH_AUDIT.md
```

**Gate A3:** learned memory features must outperform equal-dimensional fixed random features under the same probe and adapter.

---

## 6. Gate A — Whether large training is scientifically justified

Codex must implement a machine-readable gate job. Large-scale search may begin only when all are true for `DD-b` on switching Mackey--Glass:

```text
A1 causal deletion: PASS
A2 stability/noncollapse: PASS
A3 learned-vs-random features: PASS
```

The gate command must write:

```text
results/phase3/gates/gate_a.json
reports/phase3/GATE_A_MEMO.md
```

and exit:

```text
0   = pass, downstream jobs may run
42  = scientific gate failed, downstream jobs must not run
1   = infrastructure or analysis error
```

A failed scientific gate is a valid result, not an engineering failure.

---

## 7. Development benchmark suite

The broad overnight search is exploratory. Its purpose is to learn where persistent memory might help and to select one locked confirmatory configuration.

### Task D1 — Recurring Mackey--Glass

Use state-preserving, continuous schedules with at least four schedule families:

1. delay recurrence:

   \[
   \tau:17\rightarrow20\rightarrow17\rightarrow22\rightarrow17;
   \]

2. feedback recurrence:

   \[
   \beta:0.20\rightarrow0.22\rightarrow0.20\rightarrow0.18\rightarrow0.20;
   \]

3. mixed delay and feedback changes;
4. gradual ramps rather than abrupt changes.

Vary regime duration, transition magnitude, observation noise, and input normalization. Preserve dynamical state and delay history across parameter changes.

### Task D2 — Switching NARMA

Include NARMA-10 and NARMA-20. Vary:

- coefficient regime;
- exogenous input distribution;
- recurrence order;
- abrupt versus gradual transition;
- return to a previously observed regime.

### Task D3 — Switching Lorenz-63

Use state-preserving changes in a control parameter such as \(\rho\), with recurring and novel regimes. Evaluate one-step prediction and short rollouts. Treat this as a replication task, not a source of model-selection decisions until the implementation is numerically validated.

### Task D4 — Controlled prototype-switch system

Create a low-dimensional system with known regime-specific local predictors and recurring latent prototypes. This is a mechanism sanity test: the true support structure is known, so support recovery, causal deletion, and reacquisition can be evaluated directly.

### Conditional tasks

Do not launch TinyStories, large natural-language modeling, Burgers, or fluid experiments during the primary overnight campaign. They are conditional on a passing confirmatory dynamic-memory gate.

---

## 8. Model-scale ladder

Define scale by target trainable parameters; Codex must solve architecture widths to parameter-match variants within 0.5%.

| Scale | Target parameters | Context lengths | Support counts | Intended lane |
|---|---:|---|---|---|
| `XS` | 40k | 32, 64 | 16, 32, 64 | local and cluster |
| `S` | 250k | 64, 128 | 32, 64, 128 | local and cluster |
| `M` | 1M | 64, 128, 256 | 64, 128, 256 | cluster |
| `L` | 4M | 128, 256 | 128, 256 | cluster |
| `XL` | 16M | 256, 512 | 256, 512 | conditional cluster scale test |

Do not assume that larger is better. The scale ladder tests whether the memory effect grows, saturates, or disappears.

For each scale, record:

- actual parameter count;
- estimated forward FLOPs;
- measured training samples/s;
- batch-1 inference latency;
- saturated throughput;
- peak VRAM;
- total GPU-hours;
- validation loss versus training samples and wall time.

---

## 9. Overnight development search

### Search Stage S1 — Broad, prunable screen

Primary variants:

```text
D0, DD-b, DR-b, RF-b
```

Tasks:

```text
switching_mackey_glass
switching_narma
prototype_switch
```

Scales:

```text
XS, S, M
```

Budget:

- 18 to 24 trials per task × variant × scale cell;
- equal trial count per variant;
- two development training seeds per trial;
- three development schedules per seed;
- fidelity rungs at approximately 10%, 30%, 60%, and 100% of the sample budget;
- prune only from development metrics.

This corresponds to roughly 650--850 trial rows before pruning.

### Search Stage S2 — Long-training scale screen

For each task and variant, promote the top two feasible S1 configurations under the lexicographic rule below to:

```text
M, L
```

and optionally `XL` when VRAM and queue budget permit.

Run:

- three development seeds;
- five held-out development schedules per seed;
- 25M observed tokens/samples for `M`;
- 50M for `L`;
- 100M for `XL` if launched;
- at least 100 evaluation checkpoints over training;
- no early stopping before 30% of the budget.

### Search feasibility and ranking

A configuration is feasible only if:

1. stationary NMSE is no more than 5% worse than the matched `D0` configuration;
2. all runs are finite;
3. effective support count is at least 25% of the bank size;
4. dead/duplicate supports remain below 25%;
5. measured overhead is recorded.

Rank feasible configurations by:

1. mean late post-transition NMSE;
2. cumulative post-transition NMSE;
3. reacquisition loss when a prior regime returns;
4. latency as a tie-breaker.

Do not combine these into an opaque weighted score.

### Search parameters

Codex may refine ranges after pilot profiling, but the initial search space must cover:

```text
backbone learning rate: 1e-5 to 3e-3, log scale
weight decay: 1e-7 to 1e-2, log scale
dropout: 0.0 to 0.20
warmup fraction: 0.0 to 0.10
gradient clip: 0.5 to 5.0
context length: scale-appropriate values above
support count: scale-appropriate values above
route projection dimension: 16 to min(256, d_model)
memory temperature: 0.03 to 3.0, log scale
memory residual gate initialization: -4.0 to 0.0
memory dropout: 0.0 to 0.10
route-entropy regularization: 0, 1e-6 to 1e-3
NLMS step size: 0.005 to 1.0, log scale
NLMS epsilon: 1e-8 to 1e-2, log scale
```

The same number of search opportunities must be given to every primary variant.

### Search backend

Do **not** let hundreds of multi-node workers write concurrently to one SQLite database.

Preferred order:

1. static deterministic trial manifest using seeded Sobol or random proposals, one independent output directory per row;
2. batch-synchronous TPE where a coordinator proposes the next batch after merging the previous batch;
3. PostgreSQL/MySQL-backed Optuna if a reliable service is already available;
4. SQLite only for single-node or single-writer local work.

Preserve the existing Optuna studies, but do not make them the concurrency bottleneck for the cluster campaign.

---

## 10. Confirmatory experiment

Run only after Gate A passes and the development search locks one architecture/configuration without inspecting confirmatory streams.

### Primary comparison

```text
DD-b + NLMS  versus  D0 + NLMS
```

### Secondary comparisons

```text
DR-b + NLMS  versus  DD-b + NLMS
RF-b + NLMS  versus  DD-b + NLMS
```

### Confirmatory data

Primary task: recurring Mackey--Glass only.

Use:

- 16 **new** training seeds;
- at least 10 new held-out schedule/stream combinations per training seed;
- abrupt and gradual transitions;
- familiar-regime returns and novel regimes;
- locked schedule generator code and locked seed list;
- no hyperparameter changes after the first confirmatory job starts.

If a development-data power analysis, performed before confirmation, estimates that 16 seeds provide less than 80% power for the preregistered 15% effect, increase the fixed sample size to at most 24 seeds before any confirmatory result is examined.

### Primary endpoint

For each transition \(j\), define a registered late window after an immediate transient. Aggregate normalized squared error over that window, transitions, and schedules within training seed:

\[
L_{\mathrm{late},s}
=\frac{1}{JH}\sum_{j=1}^J\sum_{t=t_j+h_0}^{t_j+h_0+H-1}\mathrm{NMSE}_{sjt}.
\]

Set \(h_0\) and \(H\) from development data and lock them in the confirmatory config.

Primary effect:

\[
\delta_s
=\frac{L_{\mathrm{late},s}^{D0}-L_{\mathrm{late},s}^{DD-b}}
       {L_{\mathrm{late},s}^{D0}}.
\]

### Statistical plan

For the primary comparison:

- report the paired mean and median \(\delta_s\);
- paired bootstrap 95% confidence interval over training seeds;
- exact paired sign-flip/permutation test where computationally feasible;
- no multiplicity correction for the single preregistered primary hypothesis;
- report all seed-level paired differences.

For secondary comparisons, apply Holm correction.

Also perform a TOST-style equivalence analysis with a practical equivalence margin of ±5% relative late loss. This enables a strong negative conclusion when the confidence interval is narrow enough.

### Promotion gates

Promote persistent memory only if all conditions pass:

```text
stationary degradation <= 5%
mean late post-shift improvement >= 15%
primary paired 95% CI entirely above zero
primary paired test passes alpha = 0.05
causal top-vs-random deletion gate passes
learned-vs-random feature gate passes
support stability/noncollapse gate passes
parameter, latency, VRAM, and GPU-hour overhead reported
```

### Possible decisions

1. **Promote:** all gates pass and a second dynamical task replicates directionally.
2. **Task-specific:** Mackey--Glass passes but second task does not; report a narrow task-specific result.
3. **Equivalent and more expensive:** equivalence margin passes and overhead is material; stop KAM-specific work.
4. **Radial rejected:** `DR-b` is reliably worse or no better than `DD-b`; remove radial memory from future work.
5. **Inconclusive:** confidence interval remains too wide; report required seed count rather than claiming success or failure.

---

## 11. Efficiency and scaling analysis

Benchmark exact-matched `D0`, `DD-b`, `DR-b`, and `RF-b` at:

```text
sequence lengths: 32, 64, 128, 256, 512
support counts: 16, 32, 64, 128, 256, 512
batch sizes: 1 and maximum stable throughput batch
precision: FP32 and best stable AMP mode
execution: eager and torch.compile when supported
```

Measure:

- forward latency;
- forward + backward latency;
- samples/s or tokens/s;
- peak allocated and reserved VRAM;
- optimizer-step time;
- NLMS update time separately;
- checkpoint I/O time;
- optional mean GPU power and energy per million samples if obtainable reliably.

Use CUDA events or synchronized benchmarking. Include warm-up and report median, IQR, and p90 over repeated trials.

Fit descriptive empirical scaling models, without overclaiming asymptotic proof:

\[
t_{D0}(T) \approx a+bT^2,
\]

\[
t_{DD}(T,M) \approx a+bT^2+cTM,
\]

and report fitted residuals and hardware type.

Required outputs:

```text
results/phase3/efficiency/benchmark_rows.parquet
results/phase3/efficiency/scaling_fits.json
reports/phase3/EFFICIENCY_REPORT.md
reports/phase3/figures/latency_scaling.png
reports/phase3/figures/vram_scaling.png
reports/phase3/figures/quality_cost_frontier.png
```

---

## 12. Local overnight lane

Codex must create a local launcher that detects GPU model and VRAM and selects safe settings rather than hard-coding a specific card.

Default local work:

1. repository and checkpoint audit;
2. causal deletion and feature probes;
3. unit/integration tests;
4. `XS` and `S` pilot runs;
5. small static-manifest search;
6. aggregation and report generation.

Requirements:

- one active training process per GPU unless profiling proves multiple small jobs are faster;
- automatic mixed precision when stable;
- checkpoint at least every 15 minutes and on clean termination;
- hard wall-clock budget configurable through `LOCAL_OVERNIGHT_HOURS`, default 10;
- temperature, memory, and disk-space guardrails;
- log tail and progress summary written every 10 minutes;
- stop launching new rows when remaining wall-clock is less than the estimated p90 row duration;
- never delete earlier Phase II artifacts.

Create:

```text
scripts/run_phase3_local_overnight.sh
configs/phase3/local_overnight.yaml
```

---

## 13. HiPerGator execution lane

Codex must create generic SLURM scripts with placeholders rather than assuming the user’s account, QOS, or partition.

Before submission, the launcher must print and require values for:

```text
HPG_ACCOUNT
HPG_QOS
HPG_PARTITION
HPG_GROUP
HPG_BLUE_ROOT
MAX_CONCURRENT_GPUS
```

Operational rules:

- use batch jobs for overnight work;
- use one GPU per independent trial unless a scale explicitly requires distributed training;
- use job arrays backed by a static manifest;
- throttle array concurrency with `%MAX_CONCURRENT_GPUS`;
- keep all job I/O under `/blue/<group>/...`, not the home filesystem;
- request realistic CPU and RAM per task based on pilots;
- write logs with `%A` and `%a` identifiers;
- use `sinfo` and `slurmInfo` output to record available partitions and group resources;
- permit L4-class GPUs for broad parallel search and reserve very large-memory accelerators for `L/XL` scale jobs only when the allocation and queue state justify them;
- make every array row independently rerunnable;
- aggregate after arrays complete;
- submit the confirmatory array only through a gate dependency.

Create:

```text
slurm/phase3_audit_array.sbatch
slurm/phase3_search_array.sbatch
slurm/phase3_scale_array.sbatch
slurm/phase3_confirm_array.sbatch
slurm/phase3_aggregate.sbatch
slurm/phase3_gate.sbatch
scripts/submit_phase3_hpg.sh
```

The submission script must support:

```bash
./scripts/submit_phase3_hpg.sh --plan-only
./scripts/submit_phase3_hpg.sh --submit-audit
./scripts/submit_phase3_hpg.sh --submit-search-after-gate-a
./scripts/submit_phase3_hpg.sh --submit-confirm-after-gates
./scripts/submit_phase3_hpg.sh --resume-failed
```

Use Slurm dependencies so that:

```text
audit array
  -> audit aggregation
  -> Gate A
  -> development search arrays
  -> search aggregation and config lock
  -> confirmatory arrays
  -> final aggregation and decision memo
```

A scientific gate failure must prevent dependent jobs from starting.

### Default resource templates

Bulk single-GPU trial:

```text
1 node
1 task
1 GPU
4--8 CPU cores
24--64 GB RAM
12 hour walltime
```

Large scale trial:

```text
1 node
1 task
1 GPU unless profiling requires otherwise
8--16 CPU cores
64--128 GB RAM
12--24 hour walltime
```

Codex must adjust these after three pilot jobs and write the chosen values into `reports/phase3/RESOURCE_FORECAST.md`.

---

## 14. Compute budget and queue sizing

Codex must produce a pilot-based forecast before the large queue is submitted.

Minimum forecast fields:

```text
median and p90 duration by scale
median and p90 VRAM by scale
expected completion fraction under pruning
estimated total GPU-hours
estimated peak concurrent Blue-storage write rate
estimated output storage
recommended array throttle
```

Suggested development ceiling:

```text
local lane: 10--14 GPU-hours
cluster development search: 500--900 GPU-hours
scale-up screen: 200--500 GPU-hours
confirmatory stage: 250--600 GPU-hours
```

These are ceilings, not targets. Codex should reduce the queue when pilots show the same scientific resolution can be achieved with less compute.

The generated manifest may contain roughly 800--1,200 independent trial/evaluation rows. It must remain below scheduler limits and use throttled arrays.

---

## 15. Quality assurance

### Unit tests

- exact parameter matching;
- radial direct-distance versus expanded score equality;
- route/value/both intervention correctness;
- random-feature dimensions and freezing;
- prequential no-leakage ordering;
- deterministic schedule regeneration;
- manifest idempotence;
- support fingerprint matching on a synthetic permutation case;
- bootstrap and permutation tests on known toy effects;
- equivalence test on known equivalent samples;
- atomic checkpoint completion.

### Integration tests

- CPU end-to-end smoke;
- single-GPU local smoke;
- one-row Slurm smoke;
- two-row array smoke;
- interrupted/requeued row;
- failed row retry;
- aggregation with missing rows;
- gate pass and gate fail dependency behavior.

### Numerical QA

- no NaN/Inf in scores, routes, predictions, or adapter weights;
- route rows sum to one within tolerance;
- all reported parameters and metrics carry units and denominators;
- log exactly when gradient scaling skips an optimizer step;
- record gradient norms and clipping frequency;
- record support temperature/bandwidth trajectories.

---

## 16. Required reports and artifacts

Codex must finish by creating:

```text
reports/phase3/PHASE2_EVIDENCE_AUDIT.md
reports/phase3/CAUSAL_SUPPORT_AUDIT.md
reports/phase3/SUPPORT_STABILITY_AUDIT.md
reports/phase3/FEATURE_PATH_AUDIT.md
reports/phase3/RESOURCE_FORECAST.md
reports/phase3/DEVELOPMENT_SEARCH_REPORT.md
reports/phase3/CONFIRMATORY_REPORT.md
reports/phase3/EFFICIENCY_REPORT.md
reports/phase3/PHASE3_RESULTS.md
reports/phase3/PHASE3_DECISION_MEMO.md
reports/phase3/PHASE3_REPRODUCIBILITY.md
```

Machine-readable outputs:

```text
results/phase3/run_manifest.parquet
results/phase3/all_metrics.parquet
results/phase3/seed_level_primary_effects.parquet
results/phase3/deletion_curves.parquet
results/phase3/support_stability.parquet
results/phase3/frozen_probe_metrics.parquet
results/phase3/efficiency/benchmark_rows.parquet
results/phase3/gates/*.json
results/phase3/locked_confirmatory_config.yaml
results/phase3/locked_confirmatory_seeds.json
```

The final decision memo must answer exactly:

1. Does learned persistent memory outperform no memory under matched capacity, adapter, and new-seed confirmation?
2. Is any benefit at least 15% in the preregistered late post-shift endpoint?
3. Is the learned bank better than equal-dimensional random features?
4. Are top-weighted supports causally important?
5. Are support functions stable and noncollapsed across seeds?
6. Does radial memory improve or degrade dot-product memory?
7. What is the latency, VRAM, parameter, and GPU-hour cost of the best candidate?
8. Is the correct decision to promote, narrow, simplify, or stop?

---

## 17. Conditional next steps after Phase III

### If persistent memory passes

1. Replicate on switching NARMA or Lorenz with a locked configuration and new seeds.
2. Introduce controlled online support plasticity only after fixed-bank causal use is established.
3. Test a one-dimensional Burgers equation with recurring viscosity or forcing regimes.
4. Evaluate whether persistent supports represent physical regimes, local phase-space regions, or merely basis features.
5. Develop a two-timescale theory for fast readout adaptation and slow geometry learning.
6. Only then consider small natural-language scaling or larger PDE systems.

### If persistent memory fails but online adapters succeed

Retain the general learned-feature-plus-fast-adapter result. Compare NLMS/RLS adapters on conventional transformers, GRUs, state-space models, and neural operators without a KAM-specific claim.

### If radial memory fails

Remove radial memory from the main architecture. Preserve it only as an interpretable diagnostic or specialized local kernel baseline.

### If evidence remains inconclusive

Use the observed paired-difference distribution to calculate the fixed seed count required to resolve the practical 15% effect. Do not continue with open-ended hyperparameter search.

---

## 18. Codex completion contract

Codex is finished only when:

- repository changes are tested and documented;
- local and Slurm launchers are created;
- a plan-only queue preview is generated;
- pilot resource estimates are produced;
- all existing-checkpoint audits complete;
- scientific gates are machine-readable;
- the overnight manifest is generated and validated;
- the queue is resumable and independently rerunnable;
- reports distinguish development from confirmation;
- every claim is traceable to seed-level data;
- the final memo can support a strong positive, strong negative, or explicitly inconclusive conclusion without changing the rules after results are known.

The goal is not to make KAM succeed. The goal is to determine, with enough compute and sufficiently controlled evidence, whether the remaining persistent-memory hypothesis is real.
