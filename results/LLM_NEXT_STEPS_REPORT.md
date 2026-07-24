# Kernel Adaptive Memory Experiments — Request for Next-Step Advice

## Objective

Assess whether Kernel Adaptive Memory (KAM)—radial context attention plus learned persistent supports—provides a meaningful advantage over dot-product attention and conventional baselines.

## Experimental setup

- Hardware: CUDA-enabled GPU; PyTorch 2.13.0+cu130.
- Main grid: 5 variants × 3 tasks × 3 seeds (45 runs), 1,500 training steps per run.
- Seeds: 7, 17, 27.
- Main variants: `kernel-self`, `memory-only`, `kam`, `dot-transformer`, `dot-hybrid`.
- Tasks: copy language, hidden-regime grammar, Mackey–Glass regression.
- Additional runs: GRU, MLP, KLMS, Mackey–Glass delay shift, Tiny Shakespeare character modeling, timing benchmarks, copy length generalization, regime-switch validation.
- Full main-grid data: `outputs/first_grid/summary.csv`.

## Main results

Mean validation metrics across the three main-grid seeds:

| Task | Kernel-self | KAM | Dot-transformer | Dot-hybrid | Memory-only |
|---|---:|---:|---:|---:|---:|
| Copy accuracy | 99.992% | 99.994% | 100.000% | 100.000% | 5.98% |
| Regime accuracy | 95.87% | 96.07% | 97.05% | 96.97% | 6.04% |
| Mackey–Glass MSE | 9.66e-5 | 7.60e-5 | 8.39e-5 | 6.54e-5 | 4.27e-4 |

Interpretation:

1. Context attention is essential. `memory-only` is near chance on the symbolic tasks and substantially worse on Mackey–Glass.
2. KAM does not show a general accuracy advantage. Dot-transformer is best on regime grammar, and dot-hybrid is best on Mackey–Glass.
3. Persistent memory appears useful for Mackey–Glass: KAM is about 9.5% better than dot-transformer, while dot-hybrid is about 13.9% better than KAM.
4. KAM’s strongest result is online adaptation after a delay shift:
   - static readout MSE: 0.01383
   - adaptive readout MSE: 0.00768
   - reduction: 44.5% overall and 55.8% late after the shift
5. Tiny Shakespeare results favored dot-transformer:
   - KAM perplexity: 7.41
   - dot-transformer perplexity: 6.96
   - KAM runtime: 27.5 s vs 13.9 s

## Diagnostic and robustness results

- Regime support purity averaged 39.6% for KAM. There are four regimes, so this is above a naive 25% reference but not strong evidence of clean prototype discovery.
- Regime-switch validation reduced KAM accuracy to 59.8% and produced perplexity 6.15.
- Copy length generalization failed: a KAM model trained on payload length 16 achieved only 16.1% accuracy at length 32, with perplexity 48.29.
- Timing at sequence length 256: full KAM 1.42 ms/iteration and 198 MB peak memory versus dot-transformer 0.54 ms and 82 MB.

## Important limitations

- KAM has approximately 1.3× as many parameters as dot-transformer in the main grid, so comparisons are not parameter-matched.
- GRU, MLP, and KLMS supplementary baselines were mostly single-seed or used different protocols.
- Only three seeds were used; Mackey–Glass variance is materially larger than symbolic-task variance.
- Attention plots have not yet been validated with token/support deletion or masking tests.
- Tiny Shakespeare was a small 1.1 MB corpus and the run used only 3,000 steps.

## Questions for next-step advice

Please recommend the smallest rigorous next experiment set that would clarify whether KAM is worth pursuing. In particular:

1. Should the next priority be parameter-matched comparisons, better persistent-support mechanisms, or causal deletion/masking tests?
2. What ablations best distinguish the value of radial geometry from the value of simply adding persistent memory?
3. How should KAM be evaluated for online adaptation so the shift result is statistically convincing?
4. What support-utilization, regime-purity, and stability diagnostics should be added?
5. Should the next benchmark focus on longer-context copy, nonstationary time series, larger natural language data, or a new task where persistent memory is theoretically necessary?
6. What minimum number of seeds, datasets, and statistical tests would be appropriate before claiming an advantage?

## Current conclusion

The evidence supports a focused follow-up on persistent memory for nonstationary or delayed regression and on causal support-use diagnostics. It does not yet support claiming that radial KAM improves general sequence-modeling accuracy or efficiency over dot-product attention.
