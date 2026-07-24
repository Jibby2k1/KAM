# Kernel Adaptive Memory Phase II
## Codex Experiment and Engineering Brief

## 1. Purpose and decision to be made

The current results do **not** justify the broad claim that radial attention is a better transformer score. They do justify a narrower investigation:

> A fixed persistent support bank may expose useful nonlinear routing features for delayed and nonstationary prediction, while a fast linear readout may adapt efficiently after distribution shift.

Phase II must isolate three claims:

1. **Persistent-memory value:** Does a finite support bank improve prediction or adaptation beyond context attention at matched capacity and compute?
2. **Radial-memory value:** Does radial geometry improve the persistent bank beyond dot-product or cosine routing?
3. **Adaptive-feature value:** Does KAM retain an adaptation advantage when the same online readout is attached to every backbone?

The main candidate architecture is therefore:

```text
dot-product context attention
        +
radial persistent-memory attention
        +
fast adaptive readout
```

Radial self-attention remains an ablation, not the default.

## 2. Execution constraints

Target workstation:

- Ubuntu Linux
- NVIDIA RTX 4070 SUPER, 12 GB VRAM
- approximately 80 GB system RAM
- single-GPU execution

Engineering constraints:

- Keep peak allocated VRAM below 10.5 GB unless a test explicitly studies capacity limits.
- Support FP32 and AMP/BF16 or FP16.
- Every experiment must be resumable and fully reconstructable from a saved configuration.
- Never tune on confirmatory test streams.
- Preserve backward compatibility with the current checkpoints and command-line interface where practical.

## 3. Required repository-level deliverables

### D0. Reproducible experiment infrastructure

Create:

```text
configs/
  phase2/
kam/
  adaptation.py
  diagnostics.py
  stats.py
  search.py
  experiment_registry.py
  data/narma.py
  data/mqar.py
  data/dyck.py
  data/stream_schedules.py
scripts/
  run_phase2_reanalysis.py
  run_phase2_screen.py
  run_phase2_search.py
  run_phase2_confirm.py
  run_phase2_language.py
  build_phase2_report.py
reports/
  phase2/
```

Add a single orchestration command:

```bash
kam-run-suite --config configs/phase2/<suite>.yaml
```

Every run must save:

- resolved configuration;
- Git commit and dirty-state flag;
- seed and data-stream seed;
- hardware, PyTorch, CUDA, and precision metadata;
- parameter count and trainable-parameter count;
- training tokens or samples, optimizer steps, and wall-clock time;
- best, final, and adaptation checkpoints;
- scalar metrics in JSON;
- per-stream metrics in Parquet or CSV;
- raw diagnostic arrays in NPZ or Parquet;
- failure state and traceback when a run crashes.

Use one SQLite study database for hyperparameter search and one consolidated results table for all completed runs.

**Acceptance criteria**

- Interrupted suites resume without rerunning completed trials.
- Two runs with the same configuration and seed reproduce the same generated data and closely reproduce metrics under deterministic mode.
- A CPU smoke suite completes in CI.

---

### D1. Factor the architecture into independently testable choices

Refactor `PairwiseAttentionScore`, `KAMBlock`, `KAMSequenceModel`, and the model factory so context and memory use independent score configurations.

Required score modes:

```text
context_score: none | dot | cosine | radial
memory_score:  none | dot | cosine | radial
```

Required independent options:

```text
context_normalize_qk: bool
memory_normalize_qk: bool
radial_metric: isotropic | diagonal
bandwidth: fixed | learned
memory_output: residual | routes | both
route_features: raw | projected
```

Use compact model labels:

| Label | Context | Persistent memory |
|---|---|---|
| `D0` | dot | none |
| `R0` | radial | none |
| `DD` | dot | dot |
| `DR` | dot | radial |
| `RR` | radial | radial, legacy control |

Append `-v`, `-a`, or `-b` for memory values/residual, routing activations, or both. Example: `DR-a` is dot context plus radial routing features without the memory-value residual.

Implement radial scoring in the equivalent form

\[
s_{ij}=\frac{q_i^\top Dk_j}{\sigma^2}
-\frac{k_j^\top Dk_j}{2\sigma^2},
\]

where the query-only term is omitted because row-wise softmax cancels it. Retain a direct-distance implementation as a numerical reference.

**Required tests**

- Direct-distance and expanded radial scores match numerically.
- Radial attention reduces to temperature-scaled dot attention when metric norms are equal.
- Query/key normalization can be enabled independently for dot, cosine, and radial scores.
- Causal and local-window masks remain correct.
- Old checkpoints load through a compatibility adapter.

---

### D2. Capacity-matching utilities

Implement two comparison modes:

1. **Same-width:** identical depth, width, heads, and FFN size.
2. **Parameter-matched:** adjust no-memory FFN width or hidden width until total trainable parameters are within 1% of the corresponding memory model.

Also record approximate FLOPs and measured wall-clock cost. Do not claim fairness from parameter matching alone.

For direct routing features, support:

- hidden-only readout;
- routes-only readout;
- hidden + routes;
- projected routes with a fixed output dimension.

**Acceptance criteria**

- The result table includes both same-width and parameter-matched comparisons.
- Parameter matching error is at most 1% unless the exact discrete architecture makes this impossible; exceptions must be logged.

---

## 4. Phase A — Reanalyze existing checkpoints before retraining

This phase must run on all available KAM, dot-hybrid, kernel-self, dot-transformer, GRU, and supplementary checkpoints.

### A1. Branch ablations

For each checkpoint, evaluate:

- intact model;
- persistent-memory residual set to zero;
- routing features hidden from the readout;
- both memory pathways disabled;
- context branch disabled where possible;
- memory routing replaced with a uniform distribution.

### A2. Key/value perturbations

Evaluate:

- shuffle support keys across supports;
- shuffle support values while preserving keys;
- replace learned keys with fixed random keys of matched norm;
- replace learned values with fixed random values;
- permute routes independently at each sample;
- freeze routes to their validation-set mean.

### A3. Causal deletion tests

For every prediction, rank supports by routing weight and measure loss after deleting:

- top 1, 2, 4, 8, and 16 supports;
- the same number of random supports;
- bottom-ranked supports.

Report deletion curves and normalized AOPC-style summaries. Because perturbation metrics can be misleading when models have different attainable score ranges, always include random and bottom-deletion controls rather than reporting raw AOPC alone.

### A4. Frozen-feature probes

Extract frozen features and fit identical readouts to:

- hidden state only;
- routes only;
- hidden state + routes;
- random features with the same dimension;
- learned keys with shuffled values.

For regression, include ridge, SGD, NLMS, and RLS readouts. For classification/language, use a matched linear softmax readout.

### A5. Outputs

Create:

```text
reports/phase2/reanalysis_summary.md
results/phase2/reanalysis_metrics.parquet
results/phase2/deletion_curves.parquet
results/phase2/frozen_probe_metrics.parquet
```

**Gate A**

Proceed to support-mechanism development only if at least one of the following holds:

- disabling memory causes a reproducible loss increase;
- learned routes outperform matched random features;
- top-support deletion is materially more damaging than random deletion;
- routes-only or hidden+routes readouts outperform hidden-only readouts.

---

## 5. Phase B — Dataset and benchmark expansion

The benchmark suite must contain several forms of “language,” each isolating a different capability.

### B1. Context-retrieval language: MQAR

Implement multi-query associative recall:

- present multiple key-value bindings;
- interleave distractors;
- issue several later queries;
- predict the corresponding values.

Sweep:

```text
sequence length: 64, 128, 256, 512
number of bindings: 4, 8, 16, 32
number of queries: 1, 4, 8
vocabulary size: 64, 128, 256
```

Metrics:

- query-token accuracy;
- accuracy versus sequence length and number of bindings;
- context-attention deletion faithfulness;
- latency and memory.

This primarily tests context retrieval; persistent memory is not expected to be necessary.

### B2. Length-generalization language: variable copy

Repair the current copy benchmark:

- remove learned absolute positions as the only position mechanism;
- support relative bias, sinusoidal, or rotary positions;
- train on payload lengths sampled uniformly from 8 to 64;
- test at 80, 96, 128, and 192;
- include noise/distractor tokens between source and copy regions.

Metrics:

- copied-token accuracy;
- exact-sequence accuracy;
- degradation versus unseen length;
- attention deletion faithfulness.

### B3. Hierarchical formal language: bounded Dyck-2

Generate balanced strings with two bracket types and controlled maximum depth.

Train depths:

```text
1 to 8
```

Test depths:

```text
8, 10, 12, 16
```

Evaluate next-token prediction and grammatical-validity classification. This tests hierarchical state and out-of-distribution depth generalization.

### B4. Reusable-regime symbolic language

Extend the hidden-regime grammar to continuous streams with schedules such as:

```text
A -> B -> A -> C -> A
```

Include abrupt and gradual transitions. Regime labels may be retained for diagnostics but never provided to the model.

Metrics:

- prequential token loss;
- recovery time after switching;
- reacquisition time when A returns;
- adjusted mutual information between support assignments and regimes;
- support reuse versus creation or drift.

### B5. Delayed dynamical language: recurring Mackey–Glass

Generate one continuous stream while changing parameters without resetting state or history.

Default regimes:

```text
A: tau=17, beta=0.20
B: tau=20, beta=0.20
C: tau=17, beta=0.22
```

Required schedules:

```text
A -> B -> A
A -> B -> A -> C -> A
A -> gradual(B) -> A
```

Use a history buffer large enough for the maximum delay. Validate that parameter switching is applied causally and that the stream does not restart at boundaries.

### B6. Second delayed system: switching NARMA

Implement stable NARMA-10 and NARMA-20 generators with bounded random inputs. Validate every generated stream for finite values and reject unstable parameter draws.

Use recurring schedules that change:

- NARMA order;
- nonlinear coefficient strength;
- input distribution amplitude;
- observation noise.

Required schedules:

```text
N10-A -> N10-B -> N10-A
N10 -> N20 -> N10
clean -> noisy -> clean
```

Metrics mirror Mackey–Glass, using NMSE/NRMSE in addition to MSE.

### B7. Optional third physical system: Lorenz-63

Add only after Mackey–Glass and NARMA are stable. Use parameter changes such as recurring values of \(\rho\), preserve state across switches, and evaluate one-step and short-rollout prediction.

### B8. Conditional natural-language stage: TinyStories

Run only after the persistent-memory and adaptation gates pass.

Use token budgets rather than epochs:

```text
screening: 10 million tokens
confirmation: 50 million tokens
```

Target models should remain approximately 2–15 million parameters so they are practical on the 12 GB GPU. Compare `D0`, `DD`, and `DR`; do not spend the first natural-language budget on `RR` unless radial self-attention survives earlier ablations.

Metrics:

- validation cross-entropy and perplexity;
- training tokens per second;
- peak VRAM;
- MQAR-style recall probes generated from the same vocabulary;
- memory-route utilization and deletion faithfulness.

---

## 6. Phase C — Proper online-adaptation evaluation

Implement a strictly prequential loop:

```text
predict -> score -> reveal target -> update
```

No adapter may use a target before its prediction is recorded.

### C1. Apply the same adapters to every backbone

Required conditions:

1. frozen model;
2. adaptive linear readout with NLMS;
3. adaptive linear readout with SGD;
4. adaptive linear readout with RLS;
5. optional memory-value-only update;
6. optional support/key update, only after the fixed-geometry tests pass.

Apply these to `D0`, `R0`, `DD`, `DR`, `RR`, GRU, and the fixed-budget KLMS baseline wherever the output type permits.

### C2. Primary adaptation metrics

For a shift at \(t_0\), report:

\[
L_H = \sum_{t=t_0}^{t_0+H-1}\ell_t,
\]

plus, where an oracle is available,

\[
L_H^{\mathrm{excess}}
=\sum_{t=t_0}^{t_0+H-1}
\left(\ell_t-\ell_t^{\mathrm{oracle}}\right).
\]

Also report:

- immediate post-shift loss;
- peak loss;
- recovery time to within 5% and 10% of the new steady-state loss;
- late post-shift loss;
- reacquisition time when a previous regime returns;
- forgetting on earlier regimes;
- update FLOPs, latency, and state memory.

### C3. Development and confirmatory schedules

Separate schedule sets:

- **development schedules:** used for hyperparameter tuning;
- **confirmatory schedules:** held out until final evaluation;
- **novel-regime schedules:** contain parameter values not seen during training or tuning.

**Primary confirmatory comparisons**

1. `DD + NLMS` versus `D0 + NLMS`: persistent-memory value.
2. `DR + NLMS` versus `DD + NLMS`: radial-memory value.
3. `DR + NLMS` versus the best GRU/transformer feature extractor with NLMS: architecture-specific adaptation value.

---

## 7. Phase D — Support diagnostics and interpretability

### D1. Utilization

For mean support mass \(\bar a_i\), report:

\[
N_{\mathrm{global}}=\frac{1}{\sum_i\bar a_i^2},
\qquad
N_{\mathrm{local}}(t)=\frac{1}{\sum_i a_{ti}^2}.
\]

Also record:

- top-1 selection frequency;
- dead-support fraction;
- usage Gini coefficient;
- routing entropy;
- utilization by regime and by error quantile.

### D2. Collapse and redundancy

Report:

- pairwise key distances in the learned metric;
- nearest-neighbor distance per support;
- duplicate-support fraction at several thresholds;
- effective rank of the support Gram matrix;
- value-vector cosine similarity;
- metric condition number;
- learned bandwidth distribution and saturation.

### D3. Regime alignment

Use:

- adjusted mutual information;
- normalized mutual information;
- homogeneity and completeness;
- conditional entropy of regime given support;
- permutation-based null distributions.

Do not require one support per regime. Multiple supports may legitimately represent different phase-space regions within the same regime.

### D4. Cross-seed stability

Evaluate supports on a shared anchor set and create an activation fingerprint for each support. Match fingerprints across seeds with the Hungarian algorithm.

Report:

- matched fingerprint correlation;
- nearest-context overlap;
- matched value or local-expert similarity;
- support-assignment agreement;
- support movement and lifetime during online experiments.

### D5. Faithfulness

Compare top, random, and bottom deletion curves for both context tokens and persistent supports. Explanations are considered causally useful only when top-ranked deletions produce reliably greater degradation than controls.

---

## 8. Phase E — Hyperparameter search

Use Optuna with SQLite storage and pruning. A median or successive-halving pruner may stop trials only after a warm-up period and enough completed reference trials exist.

### E1. Search space

```text
d_model:              48, 64, 96, 128
num_layers:           1, 2, 3, 4
num_heads:            2, 4, 8 where divisible
num_supports:         16, 32, 64, 128, 256
context_window:       16, 32, 64, full
ffn_expansion:        2, 4, 6
learning_rate:        log-uniform 1e-4 to 3e-3
weight_decay:         log-uniform 1e-6 to 1e-2
dropout:              0.0, 0.05, 0.10, 0.20
bandwidth_init:       0.25, 0.5, 1.0, 2.0
bandwidth_learned:    false, true
route_projection_dim: 32, 64, 128
NLMS eta:             log-uniform 1e-3 to 1.0
RLS forgetting:       0.95, 0.98, 0.99, 0.995, 1.0
```

### E2. Multi-fidelity plan

For each primary dynamic task and model family:

1. Run 32 prunable trials.
2. Evaluate intermediate validation metrics at fixed sample/token budgets.
3. Promote the top 25% to the full screening budget.
4. Re-run the best five configurations with three seeds.
5. Select one configuration per family using the development schedules only.

Primary tuned families:

```text
D0, DD, DR, RR
```

Tune `R0` only as a radial-context diagnostic.

### E3. Hardware budget controls

The scheduler must support:

```text
--max-gpu-hours
--max-trials
--max-vram-gb
--resume
--stop-on-nan-rate
```

Recommended caps:

```text
reanalysis:          no training; complete all checkpoints
screening:           <= 12 GPU-hours
hyperparameter scan: <= 24 GPU-hours per dynamic task
confirmation:        <= 24 GPU-hours total per task
natural language:    <= 48 GPU-hours, conditional
```

These are resource ceilings, not required consumption targets.

---

## 9. Phase F — Statistical protocol

### F1. Screening standard

- five training seeds for shortlisted comparisons;
- at least five independently generated test streams per trained model;
- paired evaluation on identical streams;
- report every seed, not only mean and standard deviation.

### F2. Confirmatory standard

- ten training seeds;
- Mackey–Glass and switching NARMA as two independent dynamical families;
- at least three held-out shift schedules or magnitudes per family;
- multiple test streams nested under each trained seed.

Aggregate streams within a seed or use a hierarchical bootstrap. Do not treat many streams from one checkpoint as independent training replicates.

### F3. Required inference

For each preregistered primary comparison:

- paired effect per seed;
- 95% paired hierarchical-bootstrap confidence interval;
- exact or Monte-Carlo paired permutation test;
- raw distributions and standardized effect size;
- Holm correction across multiple declared primary comparisons.

### F4. Primary success thresholds

A KAM-derived model is worth continued development only if it satisfies a coherent subset of the following:

- stationary error no more than 5% worse than the strongest matched baseline;
- cumulative post-shift or excess loss at least 15% lower;
- 95% confidence interval excludes zero improvement;
- top-support deletion is more damaging than random deletion;
- support bank is neither mostly dead nor strongly redundant;
- measured latency or memory overhead is explicitly justified by accuracy, adaptation, or interpretability.

---

## 10. Phase G — Efficiency and timing

### G1. Correct benchmark pairs

Benchmark:

```text
D0 vs R0
DD vs DR vs RR
D0 vs DD
R0 vs RR
```

This separates context-score cost, memory-score cost, and persistent-bank cost.

### G2. Measurement protocol

Use `torch.utils.benchmark` or equivalent synchronized CUDA timing with:

- warm-up iterations;
- repeated measurements;
- median, interquartile range, and p90;
- eager and `torch.compile` modes;
- FP32 and AMP;
- forward-only, forward+backward, and incremental inference;
- fixed-batch comparisons and separately reported maximum-throughput comparisons.

Sweep:

```text
sequence length T: 32, 64, 128, 256, 512, 1024 where memory permits
support count M:   16, 32, 64, 128, 256
batch size:        1, 8, 32, and maximum feasible
```

Report:

- latency;
- tokens/s or samples/s;
- peak allocated and reserved VRAM;
- parameter count;
- estimated FLOPs;
- compilation time separately from steady-state timing.

### G3. Optimization path

Implement and compare:

1. current direct radial implementation;
2. expanded bilinear-plus-key-bias implementation;
3. compiled implementation;
4. optional FlexAttention score modification where supported.

Do not optimize before numerical-equivalence tests pass.

---

## 11. Required run matrices

### Matrix 1 — Existing-checkpoint diagnostics

All available checkpoints, all perturbations, no retraining.

### Matrix 2 — Dynamic mechanism screen

```text
models: D0, R0, DD-v, DD-a, DD-b, DR-v, DR-a, DR-b, RR-b
tasks: recurring Mackey-Glass, switching NARMA
seeds: 5
```

Nominal total: 90 training runs before pruning or search.

### Matrix 3 — Formal and retrieval languages

```text
models: D0, R0, DD, DR, RR
tasks: MQAR, variable copy, bounded Dyck-2, switching grammar
seeds: 5
```

Nominal total: 100 training runs, with several length/depth evaluations performed from each checkpoint.

### Matrix 4 — Hyperparameter search

```text
families: D0, DD, DR, RR
tasks: recurring Mackey-Glass, switching NARMA
trials: 32 prunable trials per family-task pair
```

Nominal total: 256 trials, many expected to terminate early.

### Matrix 5 — Confirmatory adaptation

```text
models: best D0, DD, DR, best recurrent baseline
adapters: frozen, NLMS, SGD, RLS
families: Mackey-Glass, NARMA
training seeds: 10
held-out schedules: >= 3 per family
streams per checkpoint-schedule: >= 5
```

### Matrix 6 — Conditional natural language

```text
models: D0, DD, DR
token budgets: 10M, 50M
seeds: 3
```

Nominal total: 18 runs. Execute only after the dynamic-memory gate passes.

---

## 12. Automated reports

`build_phase2_report.py` must generate:

```text
reports/phase2/PHASE2_RESULTS.md
reports/phase2/PHASE2_DECISION_MEMO.md
reports/phase2/figures/
```

The results report must include:

- architecture and parameter table;
- stationary metrics;
- prequential adaptation curves;
- reacquisition and forgetting tables;
- score-geometry ablations;
- memory residual versus route-feature ablations;
- support utilization, collapse, and stability diagnostics;
- deletion-faithfulness curves;
- timing and VRAM scaling;
- per-seed distributions and statistical tests;
- all failed or pruned runs.

The decision memo must answer only:

1. Does persistent memory survive capacity and compute controls?
2. Does radial persistent routing beat dot or cosine persistent routing?
3. Does the adaptive advantage remain when every backbone has the same adapter?
4. Are the support explanations causal and stable?
5. Which architecture, if any, advances to the next phase?

---

## 13. Stop/go gates and subsequent objectives

### Stop or simplify KAM when

- persistent memory fails to outperform matched no-memory controls;
- random route features perform as well as learned supports;
- the adaptive advantage disappears under matched adapters;
- supports are unused, redundant, or causally irrelevant;
- radial memory is slower without measurable benefit.

In that case, retain only the useful component—for example, dot context plus a generic adaptive linear head.

### Advance to support plasticity when

- persistent memory provides a reproducible advantage;
- learned supports outperform random features;
- deletion tests verify causal use;
- recurring-regime reacquisition is better than controls.

Then implement:

1. stable/plastic support partitions;
2. residual-driven support replacement;
3. repulsion and utilization regularizers;
4. support merge/split operations;
5. coefficient transport after support or geometry motion.

### Advance to scaling when

- the dynamic and mechanism benchmarks pass;
- radial-memory overhead is acceptable or optimized;
- support behavior remains interpretable across seeds.

Then pursue:

1. larger TinyStories or WikiText-scale language modeling;
2. sparse or compactly supported memory routing;
3. longer contexts and cached incremental inference;
4. Burgers and latent PDE rollouts;
5. patchwise or coarse-grid fluid prediction using physical rollout metrics.

## 14. Codex completion definition

Codex should consider Phase II complete only when:

- all required code paths and tests are merged;
- existing results are reproducible;
- all six experiment matrices are either completed or explicitly stopped by a documented gate;
- the consolidated result tables and automated reports are generated;
- every primary claim is accompanied by matched controls and uncertainty;
- the decision memo selects one next architecture or recommends stopping the KAM-specific direction.

## Research basis

The evaluation design follows the prequential principle of scoring before updating; uses recurring concept schedules to test retention and reacquisition; adopts MQAR as a focused test of in-context associative recall; uses TinyStories only as a small-model natural-language stage; treats long-context/formal tasks as mechanism tests rather than general proof; and requires perturbation controls for attribution faithfulness. Performance work should use synchronized PyTorch benchmarking, compilation as a separately reported mode, and custom score optimization only after mathematical equivalence is verified.
