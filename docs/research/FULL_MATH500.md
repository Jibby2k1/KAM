# Full MATH-500 with a neutral answer prompt

This study reuses the same parameter-matched KAM and causal dot-attention adapters with the weights chosen before the earlier two-marker test. SmolLM2-1.7B-Instruct uses KL 4 and UF-hosted Llama 3 8B Instruct uses KL 0.5. All adapters use learning rate 0.00008, the same training exposure and hardware within seed, and seeds 731201, 731211, and 731221. Each backbone has an untouched reference. [Development selection and integer-subset results](MATH_KL_TRANSFER.md) · [Locked research plan](RESEARCH_PLAN.md)

## Evaluation

The study covers all 500 questions in the pinned release, including the 311 integer-answer questions from the earlier evaluation and 189 additional questions. The same instruction follows each original problem: “Solve the problem. Explain your reasoning and state your final answer clearly.” Each checkpoint uses its backbone's chat template, length-sorted batches of 64, and greedy generation stopping at EOS or 1,024 new tokens. Generation uses no content-based stop. All checkpoints were evaluated on HiPerGator RTX PRO 6000 GPUs. [Pinned MATH-500 release](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl)

One pinned Math-Verify configuration grades every completion. The extractor selects the first ranked match and accepts unmarked arithmetic, LaTeX and conventional answer cues. The same verified expression can be written as a decimal, fraction, simplified radical, polynomial or other equivalent form. The grader uses strict variable matching, permits interval-to-inequality equivalence, and scores failed extractions as incorrect. All 500 gold answers parsed and verified against themselves. Twelve authored cases passed, including equivalent expressions and wrong endpoints. [Math-Verify implementation](https://github.com/huggingface/Math-Verify) · [Gold coverage and authored cases](../../results/general_benchmarks_full_math500/gold-audit.json)

The full fixture had no exact normalized prompt matches to the audited adapter-training text, earlier GSM8K splits, or SVAMP. This audit covers the named corpora and catches exact normalized matches. Public questions can appear in backbone pretraining or related corpora. [Training-source overlap](../../results/general_benchmarks_full_math500/source-overlap.json) · [Prior fixture overlap](../../results/general_benchmarks_full_math500/prior-fixture-overlap.json)

## Results

Scores are means across the three seeds and use the full 1,024-token cap. The 95% paired item intervals condition on the measured seeds.

| Backbone | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, KL 4 | 22.80% | 23.53% | 22.80% | +0.73 [−1.40, +2.87] |
| Llama 3, KL 0.5 | 29.40% | 27.67% | 28.13% | −0.47 [−3.20, +2.27] |

SmolLM2's KAM–dot differences by seed were +0.20, −0.60, and +2.60 points. Llama's were −1.40, −0.60, and +0.60. Both full-set intervals span zero. [SmolLM2 summary](../../results/general_benchmarks_full_math500/smollm2-summary.json) · [Llama summary](../../results/general_benchmarks_full_math500/llama3-summary.json)

### Integer and noninteger answers

| Backbone and subset | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2, 311 integer | 25.72% | 25.51% | 24.76% | +0.75 [−1.82, +3.43] |
| SmolLM2, 189 noninteger | 17.99% | 20.28% | 19.58% | +0.71 [−3.00, +4.23] |
| Llama 3, 311 integer | 33.76% | 30.01% | 32.69% | −2.68 [−6.32, +0.96] |
| Llama 3, 189 noninteger | 22.22% | 23.81% | 20.63% | +3.17 [−0.71, +7.06] |

On Llama, dot led on the integer subset in all three seeds and KAM led on the noninteger subset in all three. Answer type is correlated with the kinds of problems in each subset. Treat this pattern as a replication target.

### Parser and budget sensitivity

Grading only the final three nonempty lines changed three correctness flags across 7,000 complete outputs. Full grading recorded zero parser or comparison timeouts, and the 384-token prefixes recorded two comparison timeouts. Replaying with those failures recorded explicitly reproduced every correctness flag.

A masked review checked ten hash-selected excerpts whose status changed with the budget or whose capped outputs got credit. The four sampled gains contained an explicit final answer in the longer output. In the four sampled losses, the short prefix lacked a clear final answer. The two capped successes also lacked a clear final answer. In a conservative sensitivity check that gives all cap hits zero credit, the KAM–dot gaps are +0.80 points on SmolLM2 and −0.33 on Llama. This keeps the full denominator and treats any unfinished response as incorrect.

On the same generated trajectories, scoring only the first 384 tokens gave KAM/dot **21.73% / 20.27%** on SmolLM2 and **27.33% / 27.07%** on Llama. The long cap helped both dot adapters more than their matched KAM counterparts. On Llama, 23.93% of KAM and 14.07% of dot completions hit 1,024 tokens. Full generation averaged 501.77 and 499.65 seconds per checkpoint, including prompt prefill and decoding in fixed batches and excluding queue time, model load, and CPU grading.

## Reproduce

Use the [GPU driver](../../slurm/kam_full_math500.sbatch) and [generation script](../../scripts/eval_kam_full_math500.py) with the pinned MATH-500 file and each model's existing selection record. The two model arrays each contain six fitted checkpoints and one untouched reference. Grade the outputs independently with the [symbolic grader](../../scripts/grade_kam_full_math500.py). Its environment check pins the parser, symbolic engine and dependencies. The [summarizer](../../scripts/summarize_kam_full_math500.py) validates the selected record, original checkpoint metadata, ordered fixture IDs, output budget, and matched GPU. It can replay the summaries from the numeric per-checkpoint outcomes. [SmolLM2 outcomes](../../results/general_benchmarks_full_math500/smollm2-outcomes.json.gz) · [Llama outcomes](../../results/general_benchmarks_full_math500/llama3-outcomes.json.gz) · [SmolLM2 scheduler receipt](../../results/general_benchmarks_full_math500/smollm2-scheduler.tsv) · [Llama scheduler receipt](../../results/general_benchmarks_full_math500/llama3-scheduler.tsv)
