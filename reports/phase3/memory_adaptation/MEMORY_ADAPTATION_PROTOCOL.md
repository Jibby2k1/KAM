# Phase III Memory-Adaptation Protocol

## What the original code did

The original `DD-b` protocol used step-based training rather than epochs. Its learned memory keys and values remained trainable on every optimizer step through the end of training. It had no memory freeze boundary and did not record memory-bank drift. The main training loop selected checkpoints using validation MSE; the continuous-task test split existed in the data objects but was not evaluated by the runner. The separate prequential diagnostic froze the trained backbone and memory and adapted only the readout with NLMS, so it did not test memory-bank adaptation.

## New staged protocol

`DD-b-staged` keeps the same architecture and parameter count as `DD-b`. All key/value support-bank parameters are trainable for the first 75% of training steps. At that boundary, only the key/value bank is frozen; the surrounding query, output, score, and readout parameters may continue tuning. The runner records train, validation, and held-out test MSE, global bank drift, per-support drift, attention usage, entropy, and effective support count. Held-out test results are computed after tuning and are not used for checkpoint selection.

## HPG campaign

- Jobs: search array `37955698`, aggregation `37955699`
- Rows: 96
- Completed: 96
- Run failures: 0
- Pairing: 3 tasks × 2 scales × 2 trials × 2 seeds
- Comparison: joint `DD-b` versus `DD-b-staged`

Positive effects below favor staged freezing. Each cell has 4 paired runs and is descriptive, not confirmatory.

| Task | Scale | Validation improvement | Held-out test improvement |
|---|---:|---:|---:|
| prototype switch | S | −70.3% [−180.1%, 23.1%] | −75.1% [−251.3%, 29.9%] |
| prototype switch | M | +14.6% [8.2%, 21.0%] | +12.4% [−10.9%, 26.9%] |
| switching Mackey–Glass | S | +4.9% [−5.4%, 19.0%] | +3.3% [−10.9%, 16.8%] |
| switching Mackey–Glass | M | −8.8% [−39.4%, 21.8%] | −9.7% [−41.1%, 21.7%] |
| switching NARMA | S | +9.4% [0.0%, 24.5%] | +8.9% [−1.2%, 23.7%] |
| switching NARMA | M | +6.7% [−5.2%, 25.7%] | +8.6% [−2.8%, 26.5%] |

## What the traces show

Across 24 staged runs, the memory bank moved during the adaptation phase and then stopped exactly at the scheduled boundary. The maximum pre-freeze relative drift from initialization was 14.4%. At the final pre-freeze trace point, mean movement between recorded trace points was 0.48% for keys and 0.24% for values; it was small but not zero. After freezing, the maximum key and value step deltas were both exactly 0.

For comparison, continuously trained `DD-b` still moved at the end: mean final trace-interval movement was 0.64% for keys and 0.31% for values. Its mean final drift from initialization was 9.96% for keys and 3.37% for values.

## Interpretation

The original assumption was false: `DD-b` was not implicitly becoming frozen. The new protocol successfully separates memory acquisition from final backbone/readout tuning and measures the separation directly. However, staged freezing did not produce a consistent held-out performance advantage across tasks and scales. It is therefore a valid diagnostic protocol, not yet a generally superior training rule.

Figures are in [`figures/`](figures/):

- [`memory_bank_drift.png`](figures/memory_bank_drift.png)
- [`memory_support_adaptation.png`](figures/memory_support_adaptation.png)
- [`memory_train_validation_test.png`](figures/memory_train_validation_test.png)
- [`staged_vs_joint_effects.png`](figures/staged_vs_joint_effects.png)
