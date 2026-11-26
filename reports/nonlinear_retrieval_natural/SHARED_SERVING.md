# Warm-request timing for shared relation features

We compared independent and shared exact relation extraction while running both the broad and expanded scorers. Each request started from a sealed initial pool and included fetching its text, query encoding, conditioned corpus expansion, shortlist processing, evidence selection, and reader generation for the two arms. Identical reader prompts within a request were deduplicated. Index and model initialization, the original corpus search that supplied the sealed pool, and external service overhead were excluded.

The benchmark sampled 128 questions from the archived expansion-guard cohort, with 32 per shard. Both paths ran on the same device for every question, CUDA was synchronized at the measurement boundary, and implementation order alternated within each shard. The runner required exact equality of evidence pairs, guard features, and reader exactness between timed implementations. Both profiles report the same 5,000-sample paired bootstrap of the request-level median savings.

## First-pass cache effect

Array **52950098** ran on four HPG B200 GPUs. Aggregate p50 was 104.79 ms for independent extraction and 96.94 ms for shared extraction. The paired median saving was **−19.54 ms**, with a **−42.78 to +55.92 ms** interval. Order dominated. Starting with the independent path gave a 118.82 ms paired median saving, while starting with the shared path gave −102.59 ms. The second pass benefited from repeating the same question, and this run did not establish a request-latency improvement.

## Primed-request replay

Array **52950306** repeated the same sample on four B200 GPUs and ran an untimed pass on each question before either measured path. Median request latency fell from **70.89 to 66.48 ms**, a **4.37 ms paired median saving** (95% interval **4.01 to 4.78 ms**). Shared extraction was faster on 121 of 128 questions. Both order groups showed a positive median saving. The p95 estimate rose from **98.00 to 110.69 ms**. This supports a median improvement in the tested warm dual-scorer path. It leaves the tail uncertain and does not establish a first-touch or service-level gain.

Evidence selections and reader exactness matched between both implementations and the archived batched run on all 128 questions. The conditioned search used an affinity anchor. The [broad-ranked anchor](BROAD_RANKED_ANCHOR.md) changes the dependency order and was evaluated separately for accuracy.

## Reproduce

The [Slurm driver](../../slurm/nonlinear_retrieval_shared_serving.sbatch) supports `first_pass` and `primed` modes. The [replays and statistics](../../results/nonlinear_retrieval_natural/shared_serving_v1/) pair the exact same questions within each timing experiment. Recompute the summaries using the [analysis script](../../scripts/analyze_kam_shared_serving.py) with the archived expansion-guard fixture and results.
