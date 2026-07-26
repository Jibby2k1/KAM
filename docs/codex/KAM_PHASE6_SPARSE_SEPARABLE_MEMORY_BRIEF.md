# KAM Phase VI — Sparse Separable Memory
## Codex Architecture and HiPerGator Execution Brief

Repository: `Jibby2k1/KAM`  
Target branch: current `main`  
Reference repository state when this brief was written: commit `2541e09d5dfd37ad162756cf54429f1913b27e0f`

Place this file at:

```text
docs/codex/KAM_PHASE6_SPARSE_SEPARABLE_MEMORY_BRIEF.md
```

This document is the authoritative implementation and experiment contract for Phase VI.

---

# 1. Scientific objective

The broad claim that the current dense KAM bank should replace Transformer attention is not supported.

The strongest remaining hypothesis is narrower:

> A Transformer or sequence model may benefit from a sparse support-derived memory whose algebraic values or local experts can be solved or adapted rapidly while its support geometry changes slowly and conservatively.

Phase VI must answer four questions.

1. Does sparse support memory outperform an equally costly dense FFN, learned memory tokens, MoE, or product-key memory?
2. Are learned keys better than fixed random or data-centered keys?
3. Does separating the nonlinear geometry from the linear or locally linear algebra improve optimization?
4. Is the main value static prediction, conditional compute, or fast online adaptation?

The campaign must be designed so a negative conclusion is equally useful.

---

# 2. Architectural target

## 2.1 Baseline Transformer block

Implement a modern decoder/block baseline:

\[
u =
h+
\operatorname{SelfAttn}
\left(
\operatorname{Norm}(h)
\right).
\]

The baseline feed-forward path is

\[
h^+
=
u+
\operatorname{FFN}
\left(
\operatorname{Norm}(u)
\right).
\]

For language models, use the same normalization, positional encoding, attention implementation, tokenizer, optimizer, and training-token budget for every architecture.

## 2.2 Sparse Separable KAM layer

Define

\[
z=f_\psi(\operatorname{Norm}(u)),
\]

with memory keys

\[
\mathcal K=\{k_i\}_{i=1}^{M}.
\]

A router scores the keys:

\[
s_i(z)
=
-\frac{1}{2\tau}
(z-k_i)^\top D(z-k_i),
\]

or a matched dot/cosine alternative.

Retrieve only the top \(K\) supports:

\[
\mathcal I(z)
=
\operatorname{TopK}
\{s_i(z)\}_{i=1}^{M},
\qquad K\ll M.
\]

Normalize over the selected supports:

\[
a_i(z)
=
\frac{\exp s_i(z)}
{\sum_{j\in\mathcal I(z)}\exp s_j(z)}.
\]

The preferred expert is locally affine:

\[
g_i(u)
=
A_i u+b_i.
\]

The memory output is

\[
m(u)
=
\sum_{i\in\mathcal I(z)}
a_i(z)g_i(u).
\]

Complete the block with a zero-initialized gate:

\[
h^+
=
u+
\operatorname{FFN}_{\mathrm{small}}
\left(
\operatorname{Norm}(u)
\right)
+
\gamma(u)W_o m(u),
\]

where the gate begins near zero.

The dense FFN must be reduced so that active FLOPs and active parameters can be matched against the baseline.

## 2.3 Required memory-value modes

Implement:

```text
vector_value
affine_expert
low_rank_affine_expert
routes_only
```

For a low-rank affine expert:

\[
g_i(u)
=
U_i(V_i^\top u)+b_i,
\]

with rank \(r_e\ll d\).

## 2.4 Required geometry modes

Implement:

```text
fixed_random
fixed_data_sample
fixed_kmeans
fixed_farthest_point
learned_full
learned_low_rank_delta
product_key
episodic_observed
```

For low-rank geometric adaptation:

\[
k_i=k_i^{(0)}+U_i v_i
\]

or use shared low-rank query/key adapters:

\[
W_Q=I+U_QV_Q^\top,
\qquad
W_K=I+U_KV_K^\top.
\]

The conservative default is one shared latent coordinate system. Fully independent query and key networks remain an ablation.

## 2.5 Global and episodic memory

Support two banks:

```text
global_bank
episodic_bank
```

The global bank is slow and persistent. The episodic bank stores recent observed key/value pairs.

Combine them with a gate:

\[
m_t
=
g_t m_t^{\mathrm{global}}
+
(1-g_t)m_t^{\mathrm{episodic}}.
\]

The episodic bank must have explicit insertion, replacement, age, and capacity policies.

---

# 3. Required Transformer and memory baselines

Implement the following common interface.

| Label | Description |
|---|---|
| `T0` | modern dense Transformer |
| `T-WIDE` | Transformer with an enlarged FFN matched to KAM total/active cost |
| `T-MEMTOK` | Transformer with persistent learned memory tokens |
| `T-MOE` | top-1/top-2 sparse mixture-of-experts FFN |
| `T-PKM` | product-key memory |
| `T-KAM-F` | sparse KAM with fixed/data-centered keys |
| `T-KAM-L` | sparse KAM with learned keys |
| `T-KAM-ALT` | KAM with alternating geometry/algebra optimization |
| `T-KAM-VP` | KAM with variable-projection-style optimization |
| `T-KAM-ONLINE` | KAM with online value/expert adaptation |
| `T-KAM-DUAL` | global plus episodic KAM memory |

For regression/dynamics, also retain:

```text
GRU
MLP
D0
budgeted_KLMS
RBF_fixed
```

No KAM result may be described as Transformer-superior unless it beats `T0`, `T-WIDE`, and at least one conditional-memory baseline at a matched resource view.

---

# 4. Code deliverables

## 4.1 Repository layout

Create:

```text
kam/phase6/
  manifest.py
  run_array.py
  aggregate.py
  gates.py
  stats.py
  diagnostics.py
  resource_forecast.py

kam/transformer/
  config.py
  block.py
  decoder.py
  normalization.py
  positional.py
  feedforward.py

kam/memory/
  interface.py
  sparse_kam.py
  routers.py
  exact_router.py
  topk_router.py
  product_key_router.py
  episodic.py
  initializers.py
  experts.py
  gates.py
  drift.py

kam/optimization/
  algebra.py
  ridge.py
  rls.py
  nlms.py
  alternating.py
  variable_projection.py
  trust_region.py
  dictionary_update.py

kam/data/phase6/
  dynamics.py
  retrieval.py
  symbolic.py
  language.py

configs/phase6/
scripts/
tests/
reports/phase6/
results/phase6/
```

Do not break historical checkpoint loading.

## 4.2 Unified memory interface

Define an interface similar to:

```python
class MemoryLayer(nn.Module):
    def route(self, hidden, *, return_diagnostics=False): ...
    def retrieve(self, hidden, routing): ...
    def update_algebra(self, features, targets, **kwargs): ...
    def update_geometry(self, loss, **kwargs): ...
    def diagnostics(self): ...
```

Routing and retrieval must be independently testable.

## 4.3 Sparse routers

Implement:

1. exact dense score plus top-\(K\);
2. chunked exact routing;
3. product-key routing;
4. optional approximate nearest-neighbor routing.

All sparse routers must have an exact-reference test on small problems.

Required router diagnostics:

```text
recall_at_k_against_exact
routing_entropy
effective_support_count
load_balance
dead_support_fraction
duplicate_fraction
tokens_per_support
```

## 4.4 Expert storage

Avoid a dense tensor of unrestricted \(M\times d\times d\) affine experts at large \(M\).

Implement:

```text
full_affine              # small M only
low_rank_per_support
shared_basis_coefficients
vector_values
```

For a shared basis:

\[
A_i
=
\sum_{\ell=1}^{R}
c_{i\ell}B_\ell.
\]

This permits many supports with manageable storage.

## 4.5 Zero-initialized residual gate

The KAM branch must initially reproduce the baseline:

```text
gate output near zero
same baseline FFN path
same initial logits/predictions within tolerance
```

Add tests verifying that enabling an untrained KAM layer does not materially change initial outputs.

## 4.6 Active resource accounting

Record:

```text
active_parameter_count
total_parameter_count
active_parameters_per_token
estimated_forward_flops
measured_forward_ms
measured_backward_ms
tokens_or_samples_per_second
peak_vram
KV_cache_bytes
memory_bank_bytes
optimizer_state_bytes
```

Report total parameters and activated parameters separately for MoE, PKM, and sparse KAM.

---

# 5. Geometry–algebra optimization

## 5.1 Parameter partition

Define:

\[
\theta =
\{\text{backbone, query map, metric, keys, router}\}
\]

and

\[
W =
\{\text{values, local experts, final algebraic readout}\}.
\]

The code must expose these groups explicitly.

## 5.2 Joint-SGD control

Retain ordinary joint training:

```text
optimizer updates theta and W every batch
```

This is the baseline optimization condition, not the recommended default.

## 5.3 Alternating optimization

Implement the loop:

```text
for outer_iteration:
    freeze geometry
    optimize/solve algebra for A steps
    freeze algebra
    update geometry for G steps
    evaluate trust-region conditions
    accept, shrink, or rollback geometry update
```

Sweep \(A:G\) ratios such as:

```text
1:1
8:1
32:1
128:1
full algebra solve : 1 geometry step
```

## 5.4 Exact algebra solve for squared loss

When the prediction is linear in algebraic parameters:

\[
\widehat Y=\Phi_\theta(X)W,
\]

solve:

\[
W^\star(\theta)
=
\left(
\Phi_\theta^\top\Phi_\theta+\lambda I
\right)^{-1}
\Phi_\theta^\top Y.
\]

Provide:

```text
direct Cholesky
conjugate gradient
blockwise ridge
streaming RLS
```

Use numerically stable regularization and report conditioning.

## 5.5 Variable projection

Implement a regression-first variable-projection path:

\[
F(\theta)
=
\mathcal L
\left(
\theta,
W^\star(\theta)
\right).
\]

Compare:

```text
stop-gradient through W*
implicit differentiation
unrolled inner solve
```

Run exact-gradient checks on tiny problems.

For language/cross-entropy, use an approximate bilevel version:

```text
inner optimization over memory values/experts
outer optimization over geometry
truncated or implicit gradient
```

Do not claim exact variable projection for nonquadratic language loss.

## 5.6 Trust-region geometry update

Before a geometry step, evaluate a fixed anchor set \(X_A\).

Define feature/function drift:

\[
\Delta_\Phi
=
\frac{
\|\Phi_{\theta'}(X_A)-\Phi_\theta(X_A)\|_F
}{
\|\Phi_\theta(X_A)\|_F+\epsilon
},
\]

\[
\Delta_f
=
\frac{
\|f_{\theta',W'}(X_A)-f_{\theta,W}(X_A)\|_F
}{
\|f_{\theta,W}(X_A)\|_F+\epsilon
}.
\]

Accept a geometry update only when:

```text
validation/replay objective improves
drift is below threshold
support utilization does not collapse
conditioning remains acceptable
```

Otherwise shrink or rollback the step.

## 5.7 Algebra transport

After a geometry move, re-solve the algebra or transport it on anchor data:

\[
W_{\mathrm{new}}
=
\arg\min_W
\|
\Phi_{\mathrm{new}}(X_A)W
-
\Phi_{\mathrm{old}}(X_A)W_{\mathrm{old}}
\|^2
+
\lambda\|W-W_{\mathrm{old}}\|^2.
\]

Record transport error.

## 5.8 Dictionary-learning control

Implement an optional support update based on:

```text
assignment / sparse codes
coverage
centroid or dictionary update
replacement of unused supports
```

Compare against gradient-based key learning.

---

# 6. Regularization

Implement independently configurable losses.

## Coverage

\[
\mathcal L_{\mathrm{cov}}
=
\mathbb E_z\min_i\|z-k_i\|_D^2.
\]

## Repulsion

\[
\mathcal L_{\mathrm{rep}}
=
\sum_{i\neq j}
\exp
\left(
-\frac{\|k_i-k_j\|_D^2}{\rho^2}
\right).
\]

## Load balance

\[
\mathcal L_{\mathrm{bal}}
=
\sum_i
\left(
\bar a_i-\frac{1}{M}
\right)^2.
\]

## Drift

\[
\mathcal L_{\mathrm{drift}}
=
\mathbb E_{x\in X_A}
\|
m_{\mathrm{new}}(x)-m_{\mathrm{old}}(x)
\|^2.
\]

## Metric conditioning

Regularize the metric or low-rank adapters so singular values remain bounded.

Every regularizer must have a zero-weight control.

---

# 7. Datasets and task lanes

## Lane A — Controlled regression and dynamics

Use:

```text
controlled prototype regimes
switching Mackey–Glass
stable switching NARMA
Lorenz-63
Rössler
optional Burgers latent dynamics
```

Vary:

```text
recurrence
regime separation
observation noise
process noise
observability
context-to-memory ratio
abrupt/gradual shifts
```

Primary metrics:

```text
global NMSE
rollout error
stationary degradation
late post-shift NMSE
reacquisition time
```

## Lane B — Retrieval and formal language

Use:

```text
MQAR
variable copy
associative recall with distractors
controlled symbolic regimes
bounded Dyck
```

These isolate retrieval, hierarchy, and regime reuse.

## Lane C — Small language modeling

Use a modern decoder baseline on:

```text
TinyStories
WikiText-103 or equivalent small corpus
```

Conditional larger stage:

```text
a fixed 0.5B–2B token clean web/text subset
```

Run only after small-scale architecture and systems gates pass.

Primary metrics:

```text
validation cross-entropy
perplexity
tokens/sec
batch-1 latency
peak VRAM
retrieval-probe accuracy
```

## Lane D — Online adaptation

Use continuous streams with:

```text
A -> B -> A -> C -> A
```

Compare identical feature extractors with:

```text
frozen
SGD readout
NLMS
RLS
memory-value update
local-expert update
geometry update
```

The loop is always:

```text
predict -> score -> reveal -> update
```

---

# 8. Parallel implementation workstreams

Codex should split implementation into concurrent workstreams.

## Workstream A — Modern Transformer baselines

Deliver:

```text
T0
T-WIDE
T-MEMTOK
tests
resource accounting
```

Independent of sparse-router implementation.

## Workstream B — Sparse memory core

Deliver:

```text
memory interface
exact/chunked top-k
keys and expert storage
zero-init gate
diagnostics
```

Depends only on the existing model abstractions.

## Workstream C — PKM and MoE baselines

Deliver:

```text
T-MOE
T-PKM
matched active-compute interface
load-balance diagnostics
```

Parallel to Workstream B.

## Workstream D — Separable optimizers

Deliver:

```text
ridge/RLS/NLMS
alternating loop
variable projection
trust region
transport
```

Can initially test on synthetic linear-feature fixtures.

## Workstream E — Data and benchmarks

Deliver all task lanes and stream-quality gates.

## Workstream F — HiPerGator infrastructure

Deliver:

```text
manifest builder
array runner
dependency gates
aggregation
statistics
resource forecast
resume logic
```

## Workstream G — Diagnostics and interpretability

Deliver deletion, fingerprints, load/utilization, drift, conditioning, and plots.

## Dependency rule

No workstream should wait for another unless its interface is required. Use temporary fixtures and mocks. Merge only after interface tests pass.

---

# 9. HiPerGator experiment program

The campaign should use staged multi-fidelity promotion rather than one indiscriminate Cartesian product.

## Stage 0 — Correctness and systems microbenchmarks

Target:

```text
100–250 jobs
```

Test:

- exact versus sparse-router outputs;
- gradient checks;
- baseline equivalence at zero gate;
- active parameter/FLOP accounting;
- router recall;
- memory scaling;
- solver correctness;
- geometry rollback.

Block all large arrays unless Stage 0 passes.

## Stage 1 — Geometry/algebra mechanism grid

Tasks:

```text
prototype
Mackey–Glass
stable NARMA
MQAR
```

Architectures:

```text
T0
T-WIDE
T-MEMTOK
T-KAM-F
T-KAM-L
```

Optimization:

```text
joint SGD
alternating 8:1
alternating 32:1
alternating 128:1
ridge re-solve
variable projection stop-grad
variable projection implicit
dictionary update
```

Geometry:

```text
fixed random
sampled data
kmeans
farthest point
learned
low-rank learned delta
```

Value mode:

```text
vector
low-rank affine
routes only
```

Supports:

```text
16, 32, 64, 128, 256, 512
```

Top-\(K\):

```text
1, 2, 4, 8, 16
```

Use a Sobol or Latin-hypercube design plus explicit controls.

Fidelities:

```text
5%, 20%, 50%, 100% budget
```

Default target:

```text
3,000–6,000 completed or pruned jobs
```

Seeds:

```text
2 at 5%
3 at 20%
5 at 50%
```

Promote by paired lower confidence bound, practical effect, and compute-normalized quality.

## Stage 2 — Transformer/memory comparison grid

Compare:

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

Model scales:

```text
2M
10M
30M
100M active parameters
```

For each scale, match:

```text
total parameters
active parameters per token
training FLOPs
inference FLOPs
training tokens/samples
```

Tasks:

```text
MQAR
controlled symbolic regimes
TinyStories
one dynamics benchmark
```

Default target:

```text
1,500–3,000 jobs
```

Do not run 100M language models until 2M–30M gates pass.

## Stage 3 — Router and systems scaling

Sweep:

```text
memory slots:
1k, 4k, 16k, 64k, 262k, optional 1M

top-k:
1, 2, 4, 8, 16, 32

routers:
exact, chunked, product-key, approximate

precision:
fp32, bf16, fp16 where safe
```

Measure:

```text
routing recall
quality
forward/backward latency
throughput
VRAM
bank storage
optimizer storage
communication cost
```

Default target:

```text
500–1,000 jobs
```

## Stage 4 — Online adaptation grid

Models:

```text
T0
T-WIDE
T-KAM-F
T-KAM-L
T-KAM-ONLINE
T-KAM-DUAL
```

Adapters:

```text
none
SGD
NLMS
RLS
value-only
expert-only
episodic insertion
slow geometry
```

Streams:

```text
Mackey–Glass schedules
NARMA schedules
prototype schedules
symbolic regimes
```

Target:

```text
1,000–2,000 jobs
```

Use five development seeds and at least ten held-out schedules per checkpoint.

## Stage 5 — Long-training scaling

Promote at most:

```text
one dense Transformer
one conventional memory baseline
two KAM variants
```

Scales:

```text
10M
30M
100M
optional 300M if strongly justified
```

Language token budgets:

```text
10M model:  200M–500M tokens
30M model:  600M–1.5B tokens
100M model: 2B–5B tokens
```

Dynamics budgets should report observations per active parameter and convergence.

Target:

```text
100–300 long jobs
```

## Stage 6 — Locked confirmation

Lock no more than three claims:

1. KAM versus widened FFN;
2. KAM versus strongest memory/MoE baseline;
3. alternating/variable-projection optimization versus joint SGD.

Use:

```text
10–16 new seeds for synthetic/dynamics
at least 3 independent language pretraining seeds
fixed held-out streams/corpora
paired or matched statistical tests
equivalence margins
```

---

# 10. Promotion and pruning

At each fidelity, prune configurations when any hold:

```text
nonfinite loss
stream-quality failure
support collapse
router recall below threshold
validation loss materially dominated
quality-per-FLOP dominated
latency or VRAM exceeds declared ceiling
geometry drift repeatedly rejected
```

Do not prune solely on early raw loss when an architecture has a known warmup phase. Use architecture-specific grace periods declared before the search.

Promote using a multi-objective frontier:

```text
quality
quality per training FLOP
quality per inference FLOP
latency
VRAM
adaptation speed
```

Store the full Pareto frontier.

---

# 11. HPG scheduling design

## Manifest strategy

Use immutable static manifests:

```text
one row = one independently rerunnable job
```

Do not use one shared SQLite writer across nodes.

Generate stage-specific manifests only after upstream gates pass.

## Array partitioning

Shard by:

```text
task
model scale
architecture family
resource class
fidelity
```

This permits appropriate wall-time and VRAM requests.

## Resource classes

Codex must profile and then define classes such as:

```text
gpu_small
gpu_medium
gpu_large
gpu_long
cpu_aggregation
```

Do not hard-code a specific HPG GPU until `sinfo` and allocation metadata are inspected.

## Concurrency

Use the highest scientifically safe concurrency allowed by the allocation, while avoiding filesystem overload.

Suggested initial throttles:

```text
small screens: 32–128 concurrent GPUs
medium runs:   16–64 concurrent GPUs
long runs:      4–16 concurrent GPUs
```

Increase only after monitoring startup, I/O, and failure rates.

## Failure handling

Retry only:

```text
preemption
node failure
filesystem failure
transient CUDA initialization failure
```

Do not retry scientific failures under a new seed silently.

---

# 12. Statistical requirements

The training seed is the inferential unit.

Aggregate evaluation streams within training seed.

Report:

```text
paired mean difference
paired relative improvement
bootstrap 95% CI
exact permutation test
effect size
equivalence test
```

Use Holm adjustment for declared families of comparisons.

For scaling, fit curves with uncertainty against:

```text
active parameters
training FLOPs
inference FLOPs
tokens/samples
GPU-hours
```

Do not infer scaling from pooled model-size means.

---

# 13. Required final artifacts

## Code

```text
kam/phase6/
kam/transformer/
kam/memory/
kam/optimization/
kam/data/phase6/
configs/phase6/
scripts/submit_phase6_hpg.sh
scripts/run_phase6_local.sh
tests/test_phase6_*.py
```

## Reports

```text
PHASE6_VALIDITY_REPORT.md
PHASE6_ARCHITECTURE_REPORT.md
PHASE6_OPTIMIZATION_REPORT.md
PHASE6_TRANSFORMER_COMPARISON.md
PHASE6_ROUTER_SCALING_REPORT.md
PHASE6_ADAPTATION_REPORT.md
PHASE6_LONG_TRAINING_REPORT.md
PHASE6_CONFIRMATORY_REPORT.md
PHASE6_DECISION_MEMO.md
PHASE6_REPRODUCIBILITY.md
```

## Machine-readable outputs

```text
run_manifest.parquet
all_metrics.parquet
paired_seed_metrics.parquet
router_metrics.parquet
geometry_drift.parquet
algebra_solver_metrics.parquet
support_diagnostics.parquet
adaptation_metrics.parquet
scaling_metrics.parquet
confirmatory_metrics.parquet
pareto_frontier.parquet
```

---

# 14. Decision outcomes

The final memo must select exactly one:

```text
PROMOTE_SPARSE_KAM_MEMORY
PROMOTE_FIXED_KEY_FAST_ALGEBRA
PROMOTE_KAM_FOR_ONLINE_ADAPTATION_ONLY
PROMOTE_CONVENTIONAL_MEMORY_BASELINE
PROMOTE_WIDENED_TRANSFORMER
RETAIN_AS_DIAGNOSTIC_ONLY
STOP_KAM_SPECIFIC_DIRECTION
```

Promotion of sparse KAM requires a clear win on at least one axis:

```text
better quality at matched compute
better quality/latency frontier
more conditional capacity at matched active FLOPs
faster and reliable online adaptation
faithful and stable support explanations
```

A result that merely adds parameters or total FLOPs is not a KAM advantage.
