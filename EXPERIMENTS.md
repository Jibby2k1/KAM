# Experimental decision table

| Experiment | Isolates | Main comparison | Success signal | Falsifier |
|---|---|---|---|---|
| Copy language | exact token retrieval | radial context vs dot context | matched accuracy with useful sparse lag maps | worse accuracy and no explanatory gain |
| Hidden-regime grammar | reusable global prototypes | hybrid vs context-only | lower loss and support/regime alignment | memory bank unused or redundant |
| Mackey-Glass | delayed dynamics and online adaptation | hybrid vs GRU, MLP, KLMS | competitive MSE and faster shift recovery | unstable geometry or no adaptation advantage |
| Character text | realistic sequence modeling | hybrid vs dot transformer | competitive perplexity at matched timing | material slowdown without accuracy or interpretability gain |

## Minimal ablation grid

For each task and seed:

1. `kernel-self`
2. `memory-only`
3. `kam`
4. `dot-transformer`
5. `dot-hybrid`
6. task-appropriate non-attention baseline

Keep model width, depth, head count, optimizer, training tokens, and evaluation budget fixed. Report parameter count and measured training time rather than comparing epochs alone.

## Diagnostics worth retaining

- row entropy and effective support count;
- support utilization and dead-support fraction;
- nearest contexts to each persistent key;
- deletion tests for high-weight context tokens;
- deletion or masking tests for high-weight persistent supports;
- stability of support assignments across seeds;
- forward latency, backward latency, throughput, and peak memory.
