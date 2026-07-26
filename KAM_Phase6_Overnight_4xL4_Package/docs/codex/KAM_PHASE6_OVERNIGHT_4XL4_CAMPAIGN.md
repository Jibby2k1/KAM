# KAM Phase VI — 4×L4 Overnight Scientific Campaign
## Codex implementation, deployment, and automatic-analysis contract

Repository: `Jibby2k1/KAM`  
Execution target: UF HiPerGator  
Concurrent allocation: exactly four NVIDIA L4 GPUs  
Minimum desired wall-clock occupancy: 10 hours  
Target wall-clock envelope: 10.5–12 hours  
Target aggregate use: 42–46 L4 GPU-hours

Place this file at:

```text
docs/codex/KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md
```

This is the authoritative task. Implement missing pieces, validate them, submit the complete dependency-gated campaign, print the job graph and artifact paths, and then stop. Do not poll jobs throughout the night. The user will inspect results in the morning.

---

# 1. Scientific objective

Phase VI infrastructure is implemented and profile-tested. The next task is real training and statistical comparison, not another infrastructure expansion.

Test whether sparse separable KAM offers a defensible advantage over modern Transformer and memory baselines on:

1. language-model quality at matched resources;
2. associative retrieval;
3. delayed nonlinear dynamics;
4. geometry–algebra optimization;
5. online adaptation;
6. router quality and systems cost.

Primary hypotheses:

- **H1 — Sparse memory versus ordinary capacity:** sparse memory beats an equally costly widened Transformer FFN.
- **H2 — Fixed versus learned geometry:** fixed/data-centered keys may match learned keys when the coordinate map is trainable.
- **H3 — Separable optimization:** fast algebra plus slow geometry beats joint SGD.
- **H4 — Conditional compute:** KAM supplies more total memory at matched active parameters/FLOPs.
- **H5 — Online adaptation:** KAM values or local experts adapt faster after regime changes.

A negative result is valid. Do not promote KAM because it has more total parameters or because a run completed.

---

# 2. Hard execution constraints

## 2.1 Saturate four L4 GPUs

Use one GPU per row and maintain four eligible rows whenever work remains:

```text
SLURM array concurrency: %4
GPU per row: 1
CPUs per row: 4–8 after profiling
RAM per row: 24–48 GB after profiling
```

Queue at least 42 GPU-hours, preferably about 44 GPU-hours. Do not create thousands of ten-second rows.

## 2.2 Time-aware budgets

Before main arrays, profile representative rows and measure:

```text
tokens/s or samples/s
seconds/step
peak VRAM
checkpoint size
evaluation overhead
```

Scale token/sample budgets to target:

```text
Wave 1 rows: 20–40 minutes
Wave 2 rows: 50–90 minutes
Wave 3 rows: 90–150 minutes
```

Minimum scientific budgets below remain binding even when throughput is high.

## 2.3 No interactive monitoring

After submission, return only:

```text
preflight job ID
Stage-1-frontier CPU job ID
Wave 1 array and aggregate/gate IDs
Wave 2 controller/array and aggregate/gate IDs
Wave 3 controller/array and final-report IDs
run root
report root
expected completion window
one status command
one report-rebuild command
```

Then stop. Do not consume tokens checking the queue overnight.

## 2.4 Reproducibility per row

Record:

```text
Git commit and dirty state
manifest hash and run ID
architecture identity
dataset/corpus checksum
training seed and data seed
precision and GPU metadata
PyTorch/CUDA versions
total and active parameters
active parameters per token
estimated and measured FLOPs
training tokens/samples
wall time and throughput
peak VRAM
best and final checkpoints
failure category
```

---

# 3. Mandatory preflight gate

Do not submit main arrays unless every check passes.

## 3.1 Tests and audits

Run:

```bash
pytest -q
```

plus Phase VI identity, finite-history, routing, schedule, parameter-match, and manifest audits.

Confirm the completed Stage 1 mechanism aggregate is readable. Regenerate its report if necessary.

## 3.2 Architecture identity

Verify distinct identities and forward behavior for:

```text
T0
T-WIDE
T-MEMTOK
T-MOE
T-PKM
T-KAM-F
T-KAM-L
T-KAM-ALT
T-KAM-VP
T-KAM-ONLINE
```

Required assertions:

- zero-gated KAM matches its baseline before memory training;
- ALT/VP report nonzero algebra and geometry phases under real schedules;
- T-WIDE uses the widened FFN;
- T-MEMTOK appends persistent tokens;
- T-MOE activates configured experts;
- T-PKM uses product-key retrieval;
- fixed KAM keys remain fixed;
- learned KAM keys receive finite gradients;
- top-k routing returns exactly k entries unless the bank is smaller.

## 3.3 Data quality

Dynamics streams must satisfy:

```text
finite_fraction == 1
minimum target variance
clip-boundary fraction below threshold
bounded amplitude
nonconstant stream
correct memory horizon
independent train/validation/test seeds
```

Language must have fixed tokenizer/corpus checksums, immutable validation data, no split overlap, and deterministic packing per seed.

## 3.4 L4 calibration

Profile:

```text
10M T0 language
10M T-KAM-F language
10M T-MOE language
1M T-KAM-ALT dynamics
```

Use observed throughput to resolve overnight budgets. Preflight failure blocks all downstream work.

---

# 4. Stage 1 frontier extraction

Before Wave 1, analyze the completed 3,000-row Stage 1 campaign on CPU.

Produce:

```text
reports/phase6/overnight/STAGE1_FRONTIER_REANALYSIS.md
results/phase6/overnight/stage1_frontier.parquet
results/phase6/overnight/stage1_pareto.parquet
```

Group by task, architecture, geometry, expert type, optimizer, router, supports, top-k, fidelity, seed, active parameters, active FLOPs, and wall time.

Required comparisons:

```text
fixed vs learned keys
vector vs affine experts
joint SGD vs ALT
joint SGD vs VP
exact vs chunked vs product-key
quality vs active FLOPs
quality vs GPU-hours
```

Automatically select at most:

```text
2 fixed-key KAM configurations
2 learned-key KAM configurations
1 ALT configuration
1 VP configuration
```

Selection requires finite metrics, valid identity, no collapse, nonzero ALT/VP geometry phases, router recall ≥0.95 when approximate, and Pareto-front relevance.

---

# 5. Compute allocation

| Work | Target GPU-hours |
|---|---:|
| Preflight/calibration | 1–2 |
| Wave 1 broad screen | 12–14 |
| Wave 2 promoted training | 16–18 |
| Wave 3 replication/adaptation | 13–15 |
| **Total** | **42–46** |

The controller may adjust by ±10% after calibration, but must not fall below 40 GPU-hours unless a scientific gate fails.

---

# 6. Wave 1 — Broad quality screen

Target wall time: 3.0–3.5 hours at four-way concurrency.  
Target use: 12–14 GPU-hours.

## 6.1 Language lane

Primary corpus: TinyStories, versioned and checksummed. If unavailable, use the largest cached legal corpus and document the substitution. Never silently use a one-sentence fallback.

Scale: about 10M total parameters.

Architectures:

```text
T0
T-WIDE
T-MEMTOK
T-MOE
T-PKM
T-KAM-F
T-KAM-L
T-KAM-ALT
T-KAM-VP
```

Use two paired seeds. Minimum 50M training tokens per row; increase to fill a 25–40 minute calibrated row.

Lock the same tokenizer, sequence length, batch-token target, RMSNorm, RoPE, SwiGLU recipe, AdamW schedule, warmup fraction, clipping, packing, and precision.

Report validation cross-entropy, perplexity, tokens/s, active FLOPs/token, total/active parameters, VRAM, batch-1 latency, and quality/GPU-hour.

## 6.2 Retrieval lane

Tasks:

```text
MQAR
associative recall with distractors
variable copy
```

Factors:

```text
sequence length: 128, 256, 512, 1024
bindings: 8, 16, 32
queries: 4, 8
distractor density: low, medium, high
```

Architectures:

```text
T0
T-MEMTOK
T-PKM
best T-KAM-F
best T-KAM-L
```

Use a constrained Sobol design and two seeds. Primary metric is query-token accuracy; secondary metrics are length degradation, latency, VRAM, and deletion faithfulness.

## 6.3 Dynamics and optimization lane

Tasks:

```text
switching Mackey–Glass
stable switching NARMA
controlled prototype regimes
Lorenz-63 or validated third fixture
```

Architectures:

```text
T0/D0
T-WIDE
T-KAM-F
T-KAM-L
T-KAM-ALT
T-KAM-VP
```

Optimization modes:

```text
joint SGD
ALT 8:1
ALT 32:1
ALT 128:1
VP stop-gradient
VP implicit if gradient checks pass
dictionary update
```

Use two screening seeds. Require at least 25 logged validation points and enough work for meaningful optimization trajectories.

Report global held-out NMSE, validation-selected held-out NMSE, rollout error where supported, algebra/geometry counts, accepted/rejected geometry steps, feature/function drift, inner-solve residual, conditioning, and support utilization.

---

# 7. Wave 1 hyperparameter search

Use a constrained Sobol pool, not a full Cartesian product.

## KAM space

```text
geometry:
  fixed_random
  sampled_data
  kmeans
  farthest_point
  learned_full
  learned_low_rank_delta

expert:
  vector
  low_rank_affine

expert_rank: 4, 8, 16, 32

memory_slots:
  language: 1024, 4096, 16384
  dynamics: 64, 256, 1024, 4096

top_k: 1, 2, 4, 8, 16
router: exact, chunked, product_key
gate_initial_logit: -8, -6, -4, -2
temperature: log-uniform 0.03–2.0
memory_lr_ratio: 0, 0.01, 0.03, 0.1, 0.3, 1.0
algebra_regularization: log-uniform 1e-6–1e-1
load_balance_weight: 0, 1e-4, 1e-3, 1e-2
coverage_weight: 0, 1e-4, 1e-3, 1e-2
repulsion_weight: 0, 1e-5, 1e-4, 1e-3
ALT ratio: 8:1, 32:1, 128:1
trust_region_drift: 0.01, 0.03, 0.1, 0.3
```

Constraints:

- product-key preferred at ≥16k slots;
- exact routing prohibited beyond profiled resource limits;
- full affine experts only in small diagnostics;
- leave at least 15% L4 VRAM safety margin.

## Transformer controls

Search a small matched set over widened FFN multiplier, memory-token count, MoE expert count/top-k, and PKM slots/top-k. Maintain total-parameter, active-parameter/token, and measured-FLOP views separately.

---

# 8. Wave 1 automatic gate

A dependent CPU job aggregates Wave 1 and writes the Wave 2 manifest.

Always retain valid runs of T0, T-WIDE, T-MEMTOK, T-MOE, and T-PKM.

Promote at most:

```text
2 fixed-key KAMs
2 learned-key KAMs
1 ALT KAM
1 VP KAM
```

Promotion is multi-objective over quality, quality/active-FLOP, quality/GPU-hour, latency, and VRAM.

Hard reject nonfinite data, invalid identity, collapse, router recall <0.95, zero geometry phases for ALT/VP, or VRAM safety violations.

Do not promote a KAM worse than T-WIDE in both quality and compute unless it has a large retrieval/adaptation advantage.

---

# 9. Wave 2 — Promoted quality training

Target wall time: 4.0–4.5 hours.  
Target use: 16–18 GPU-hours.

## 9.1 Language

Train T0, T-WIDE, T-MEMTOK, T-MOE, T-PKM, and up to six promoted KAMs.

Scales and budgets:

```text
10M models: 3 paired seeds, minimum 150M tokens
30M models: T0, T-WIDE, strongest conventional memory, top 2 KAMs;
            2 paired seeds, minimum 250M tokens
```

The controller may raise token budgets to use available time. Use fixed held-out validation and retrieval probes.

## 9.2 Dynamics

Promote the best fixed-key, learned-key, and ALT/VP KAM. Compare with T0/D0, T-WIDE, GRU, and budgeted KLMS/fixed RBF where supported.

Tasks: Mackey–Glass, stable NARMA, and best controlled regime task.

Use five paired seeds. Train until at least 100 validation checkpoints or a documented plateau. Early stopping is allowed only under registered convergence/invalidity gates.

---

# 10. Wave 2 automatic gate

Promote no more than:

```text
1 fixed-key KAM
1 learned/separable KAM
1 conventional memory baseline
T0
T-WIDE
```

Language selection uses validation quality and quality/compute. Dynamics uses paired held-out NMSE and optimization stability. Screening practical threshold: 3%.

If no KAM is within 2% of T-WIDE quality or offers a compensating systems/adaptation advantage, Wave 3 should confirm the negative result rather than invent a new model.

---

# 11. Wave 3 — Deep replication and adaptation

Target wall time: 3.5–4.0 hours.  
Target use: 13–15 GPU-hours.

## 11.1 Language replication

Models:

```text
T0
T-WIDE
strongest conventional memory
top fixed-key KAM
top learned/separable KAM
```

Use the largest budget fitting remaining time, with at least three seeds at 10M or two at 30M. Do not mix scales in paired inference.

## 11.2 Online adaptation

Use A→B→A→C→A schedules for Mackey–Glass, NARMA, prototype, and symbolic regimes.

Backbones:

```text
T0
T-WIDE
best conventional memory
best fixed-key KAM
best learned/separable KAM
```

Adapters:

```text
none
SGD readout
NLMS
RLS
KAM value/expert update
episodic insertion if validated
```

Use five development seeds and ten held-out schedules per checkpoint. Enforce predict→score→reveal→update.

Report early/late post-transition loss, cumulative excess loss, recovery time, reacquisition time, update FLOPs, and adapter-state bytes.

## 11.3 Causal diagnostics

Run top/random/bottom support deletion, key shuffle, value/expert shuffle, uniform routing, and memory-branch zeroing. Implement analogous component deletion for memory tokens, MoE, and PKM where meaningful.

---

# 12. Statistics

The training seed is the inferential unit. Aggregate held-out streams within seed.

Report paired mean and relative difference, bootstrap 95% CI, exact paired permutation test, standardized paired effect, and equivalence test. Apply Holm correction within comparison families.

Primary language comparisons:

```text
best KAM vs T0
best KAM vs T-WIDE
best KAM vs strongest conventional memory
ALT/VP vs joint-SGD KAM
```

Primary dynamics comparisons:

```text
best KAM vs D0/T0
ALT/VP vs joint SGD
KAM online adaptation vs matched Transformer adapter
```

Equivalence margins:

```text
language validation loss: ±2%
dynamics NMSE: ±5%
```

---

# 13. Required overnight artifacts

Reports:

```text
reports/phase6/overnight/OVERNIGHT_EXECUTION_REPORT.md
reports/phase6/overnight/OVERNIGHT_LANGUAGE_REPORT.md
reports/phase6/overnight/OVERNIGHT_DYNAMICS_REPORT.md
reports/phase6/overnight/OVERNIGHT_OPTIMIZATION_REPORT.md
reports/phase6/overnight/OVERNIGHT_ADAPTATION_REPORT.md
reports/phase6/overnight/OVERNIGHT_DECISION_MEMO.md
reports/phase6/overnight/OVERNIGHT_REPRODUCIBILITY.md
```

Machine-readable outputs:

```text
results/phase6/overnight/run_manifest.parquet
results/phase6/overnight/all_metrics.parquet
results/phase6/overnight/stage1_frontier.parquet
results/phase6/overnight/wave1_pareto.parquet
results/phase6/overnight/wave2_pareto.parquet
results/phase6/overnight/paired_seed_metrics.parquet
results/phase6/overnight/adaptation_metrics.parquet
results/phase6/overnight/deletion_metrics.parquet
results/phase6/overnight/resource_metrics.parquet
results/phase6/overnight/failures.parquet
```

Final decision must be exactly one:

```text
PROMOTE_SPARSE_KAM_MEMORY
PROMOTE_FIXED_KEY_FAST_ALGEBRA
PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY
PROMOTE_CONVENTIONAL_MEMORY_BASELINE
PROMOTE_WIDENED_TRANSFORMER
RETAIN_AS_DIAGNOSTIC_ONLY
STOP_KAM_SPECIFIC_DIRECTION
```

---

# 14. SLURM dependency graph

Create/update:

```text
scripts/submit_phase6_overnight_4xl4.sh
scripts/phase6_overnight_controller.py
scripts/build_phase6_overnight_report.py
```

Desired chain:

```text
preflight
 -> stage1_frontier_cpu
 -> wave1_array[%4]
 -> wave1_aggregate_gate
 -> wave2_controller_submit
 -> wave2_array[%4]
 -> wave2_aggregate_gate
 -> wave3_controller_submit
 -> wave3_array[%4]
 -> final_aggregate_report
```

Wave 2 and Wave 3 controller jobs may submit their generated arrays programmatically. All controller actions must be idempotent and recorded in:

```text
results/phase6/overnight/job_graph.json
```

Retry only infrastructure failures. Never silently retry a scientific failure with a different seed.

---

# 15. Morning handoff

After submission, Codex should return only:

1. implementation summary;
2. tests passed;
3. exact commit;
4. run/report roots;
5. SLURM job graph;
6. expected completion window;
7. one status command;
8. one report-rebuild command.

Do not poll or provide overnight chat updates.
