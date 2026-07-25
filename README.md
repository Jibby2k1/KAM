# Kernel Adaptive Memory experiments

This repository is a compact reference implementation for the architecture described in the companion paper. It is designed to answer three separate questions:

1. Does radial **context attention** retrieve useful information from the current sequence?
2. Does a fixed bank of learned **persistent supports** add value beyond context attention?
3. Can a frozen KAM feature geometry support rapid **online adaptation** through a normalized-LMS readout?

The implementation deliberately uses one generic attention calculation in two roles:

```text
Ctx: tokens query earlier tokens from the same sequence
Mem: contextual tokens query a finite learned support bank
```

All KAM variants are trained with ordinary backpropagation. The optional normalized-LMS update is applied only to the final scalar readout in the Mackey-Glass shift experiment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Model variants

| CLI name | Context attention | Persistent memory | Score |
|---|---:|---:|---|
| `kam` | yes | yes | radial energy |
| `kernel-self` | yes | no | radial energy |
| `memory-only` | no | yes | radial energy |
| `dot-transformer` | yes | no | scaled dot product |
| `dot-hybrid` | yes | yes | scaled dot product |
| `gru` | recurrent baseline | no | n/a |
| `mlp` | flattened-window baseline | no | n/a |

## Four easy experimental "languages"

### 1. Copy language: exact contextual retrieval

A random payload is shown, followed by a delimiter, and the model must emit the payload again. This primarily tests whether context attention can recover exact sequence content.

```bash
kam-train --task copy --model kam --steps 1000 --output outputs/copy_kam
kam-train --task copy --model dot-transformer --steps 1000 --output outputs/copy_dot
```

Primary metric: accuracy on the copied portion. A useful length-generalization test reserves a larger positional capacity during training and evaluates longer payloads:

```bash
kam-train --task copy --model kam --copy-length 16 --model-max-seq-len 130 \
  --steps 1000 --output outputs/copy_kam
kam-eval --checkpoint outputs/copy_kam/best_model.pt --copy-length 32 \
  --output outputs/copy_kam_length32
```

### 2. Hidden-regime grammar: persistent prototype discovery

Each sequence follows one of several second-order modular transition rules. The rule label is never supplied to the model. Persistent supports are useful only if they help identify and reuse these hidden regimes.

```bash
kam-train --task regime --model kam --seq-len 64 --steps 1500 \
  --output outputs/regime_kam
kam-train --task regime --model kernel-self --seq-len 64 --steps 1500 \
  --output outputs/regime_self
```

Primary metrics: next-token accuracy and persistent-support purity with respect to the hidden regime. Add `--regime-switch-validation` to test a rule change halfway through each validation sequence.

### 3. Mackey-Glass: delayed dynamics and online shift

Each lagged scalar is represented as a token. The model predicts the next state, then is evaluated after a change in the delay parameter.

```bash
kam-train --task mackey-glass --model kam --seq-len 32 --steps 1500 \
  --output outputs/mg_kam
kam-train --task mackey-glass --model gru --seq-len 32 --steps 1500 \
  --output outputs/mg_gru
kam-klms --window 32 --budget 128 --output outputs/mg_klms

kam-shift --checkpoint outputs/mg_kam/best_model.pt --tau 20 \
  --output outputs/mg_kam_shift
```

Primary metrics: one-step MSE, integrated post-shift error, and the early/late error of the frozen versus NLMS-adapted readout.

### 4. Character language: natural text stress test

The repository includes a tiny original text file only for smoke testing. For a meaningful experiment, provide a larger local corpus or download Tiny Shakespeare.

```bash
python scripts/download_tiny_shakespeare.py
kam-train --task char --model kam --text-path data/tinyshakespeare.txt \
  --seq-len 128 --context-window 128 --steps 3000 --output outputs/char_kam
kam-train --task char --model dot-transformer --text-path data/tinyshakespeare.txt \
  --seq-len 128 --context-window 128 --steps 3000 --output outputs/char_dot
```

Primary metrics: validation cross-entropy, perplexity, tokens per second, and context-length scaling.

## Timing and scaling

```bash
kam-benchmark --seq-lens 32 64 128 256 --output outputs/timing
kam-benchmark --seq-lens 32 64 128 --backward --output outputs/timing_train
```

The benchmark records wall-clock milliseconds, tokens per second, parameter count, and CUDA peak memory when available. Use the same device, batch size, model width, warm-up count, and precision for all comparisons.

Expected leading-order attention costs for model width `d`, sequence length `T`, local window `W`, and support count `M` are:

```text
full context attention:   O(T^2 d) time, O(T^2) attention memory
local context attention:  O(T W d) time, O(T W) attention memory
persistent memory:        O(T M d) time, O(T M) attention memory
incremental inference:    O((W + M)d) per new token with a local cache
```

The radial score uses a positive diagonal metric, so it has the same asymptotic pairwise complexity as dot-product attention. It adds norm terms and elementwise scaling, which must be measured as a constant-factor overhead.

## Outputs

Every training run writes:

- `best_model.pt`: model weights, architecture specification, and data metadata;
- `metrics.json`: training history and validation metrics;
- `context_attention.png`: final-layer context routing, when present;
- `memory_attention.png`: final-layer support routing, when present;
- `attention_diagnostics.npz`: raw arrays for custom analysis.

## Phase IV data-regime screen

The Phase IV package and execution map are summarized in [`LLM_NAVIGATION.md`](LLM_NAVIGATION.md). The authoritative scientific brief is [`KAM_Phase4_Data_Regime_Package/KAM_PHASE4_CODEX_DATA_REGIME_BRIEF.md`](KAM_Phase4_Data_Regime_Package/KAM_PHASE4_CODEX_DATA_REGIME_BRIEF.md). The bounded manifest-driven screen can be planned with:

```bash
python -m kam.phase4.manifest --config configs/phase4/factorial_screen.yaml
scripts/submit_phase4_hpg.sh --plan-only
```


## Phase V validity-gated feature campaign

The authoritative Phase V brief is [docs/codex/KAM_PHASE5_CODEX_EXECUTION_BRIEF.md](docs/codex/KAM_PHASE5_CODEX_EXECUTION_BRIEF.md), with the repository audit in [docs/codex/KAM_REPOSITORY_AUDIT_PHASE5.md](docs/codex/KAM_REPOSITORY_AUDIT_PHASE5.md). The first HPG submission is a small validity gate; the full learned-versus-fixed-feature pilot remains blocked until its machine-readable checks pass.

After the dependent HPG report job completes, read [`reports/phase5/PHASE5_VALIDITY_AUDIT.md`](reports/phase5/PHASE5_VALIDITY_AUDIT.md) for the gate result, [`reports/phase5/PHASE5_LLM_HANDOFF.md`](reports/phase5/PHASE5_LLM_HANDOFF.md) for a compact request for next-step feedback, and [`reports/phase5/PHASE5_REPOSITORY_WRITEUP.md`](reports/phase5/PHASE5_REPOSITORY_WRITEUP.md) for the human-facing summary.

## Recommended first grid

Run all five matched attention variants on the copy, regime, and Mackey-Glass tasks with three random seeds. Compare both parameter count and measured wall-clock cost. Do not interpret attractive attention plots as evidence until deletion or perturbation tests show that the attributed tokens and supports causally affect the prediction.

## Scope of this reference implementation

The code intentionally omits support birth/death, coefficient transport after geometry motion, compactly supported kernels, low-rank full metrics, and PDE data loaders. Those are follow-on experiments after the minimal hybrid demonstrates a measurable advantage.

## Phase V Stage 2 controlled-regime campaign

The Stage 2 specification is [KAM_PHASE5_STAGE2_CODEX_BRIEF.md](docs/codex/KAM_PHASE5_STAGE2_CODEX_BRIEF.md). It defines the component, capacity-crossover, factorial, and symbolic sub-studies, with independent held-out streams and executable validity gates. Inspect the plan before HPG submission:

```bash
scripts/submit_phase5_stage2_hpg.sh --plan-only
scripts/submit_phase5_stage2_hpg.sh --profile   # four 3-row HPG pre-profiles
scripts/submit_phase5_stage2_hpg.sh --submit    # only after profile review
```

Manifests are under `results/phase5/stage2/manifests/`; per-stage outputs and reports are written under `results/phase5/stage2/` and `reports/phase5/`. The nominal scale labels remain visible in each row, while `target_active_parameters` records the exact resolved architecture count used by the <=1% gate.
