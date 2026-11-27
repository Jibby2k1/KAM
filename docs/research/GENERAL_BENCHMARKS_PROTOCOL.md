# General benchmarks on the Phase VI TinyStories checkpoints

This comparison uses the first registered primary TinyStories seed, 71001, from the [Phase VI confirmation](../codex/KAM_PHASE6_CONFIRMATION_V2_PREREGISTRATION.md). It compares T-KAM-F, which adds a fixed-key KAM memory bank to each decoder layer, with T-WIDE, the registered parameter-matched dot-attention decoder. The architectures differ in how they allocate parameters. They share the TinyStories V2 128 MiB training corpus, training order seed, byte tokenizer, training token budget, and maximum input length. We used the final checkpoints and chose the seed by manifest order before looking at benchmark outcomes. Results are descriptive for this pair of checkpoints.

## HPG results

| Benchmark | KAM correct / total | Dot correct / total | KAM − dot, percentage points | Paired 95% item interval | Full initial prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 0/164 | 0/164 | +0.00 | — (all predictions scored incorrect) | 1 |
| MBPP | 0/500 | 0/500 | +0.00 | — (all predictions scored incorrect) | 432 |
| GPQA Diamond | 52/198 | 47/198 | +2.53 | [-2.02, +7.07] | 1 |
| MMLU | 3,398/14,042 | 3,454/14,042 | -0.40 | [-0.98, +0.19] | 1,233 |
| GSM8K | 0/1,319 | 0/1,319 | +0.00 | — (all predictions scored incorrect) | 3 |

Every completion on both coding tasks failed Python parsing, and neither model emitted a parseable marked numeric answer on GSM8K. These observed floor scores give no estimate of method variation. Both models strongly favored option B on the multiple-choice tasks. The two multiple-choice scores are near the four-way chance level. Only one GPQA prompt and 1,233 MMLU prompts fit in full, so those accuracy scores describe performance with mostly truncated input.

[Numerical summary](../../results/general_benchmarks/summary.json) · [Per-item numerical outcomes](../../results/general_benchmarks/paired-outcomes.json.gz)

## Evaluation protocol

Both models used the same prompt for every item and BF16 autocast on HPG L4s. The decoder can read at most 128 UTF-8 bytes. Each forward took the last 128 bytes of the available prompt and generated prefix. Its learned position indices restart at zero within each window. The results include the entire benchmark test split and record how often the complete initial prompt fits. They describe a constrained TinyStories model comparison. Published benchmark runs with different prompt, context, or scoring settings cannot be compared directly.

| Benchmark | Test split | Prompt and scoring |
| --- | ---: | --- |
| HumanEval | 164 | Author prompt, greedy continuation up to 384 bytes. Append the continuation to the original full prompt for author unit tests. Pass@1 is one generated candidate per problem. |
| MBPP | 500 | Full configuration. Zero-shot natural-language description as a Python comment, greedy generation up to 384 bytes. Score the three published tests with the supplied setup. This prompt holds the tests out of the input. |
| GPQA Diamond | 198 | Zero-shot question and four options with a fixed per-item shuffle. Accuracy uses the highest next-byte logit among A–D after the answer cue. |
| MMLU | 14,042 | All 57 test subjects, zero-shot A–D next-byte accuracy using the released option order. The headline is micro-averaged over test questions. |
| GSM8K | 1,319 | Zero-shot prompt asks for a numeric answer following four hashes. Greedy generation up to 256 bytes. Strict final-answer EM requires that marker and the exact number after comma removal. The summary records format compliance separately. |

Generation stops early on a NUL byte or three consecutive line breaks. The grader checks for a parsable program with a top-level definition before execution. The published tests run in an isolated Python 3.11.14 Apptainer container within a separate network namespace with resource and wall-time limits. An upstream reference completion for each coding benchmark must pass the grader preflight. Per-item completions and dataset text stay on HPG.

The original GPU array was 53030336. The reference generation shards ran in 53031399 and the optimized runs in 53031994. Skipping unused routing statistics reproduced all tested next-byte logits and short continuation prefixes. It also reproduced all 164 full HumanEval completions and all 256 MBPP completions from the two completed reference shards. The HumanEval logits check took 0.840 seconds with the archived evaluator and 0.018 seconds with the statistics omitted for three forwards. This timing measures evaluator overhead. The scoring runs used HPG CPU arrays 53031400, 53032417 and 53032833.

## Reproduce

The [asset fetcher](../../scripts/fetch_kam_general_benchmarks.sh) downloads pinned test assets and verifies SHA-256 hashes. GPQA uses the [public OpenAI simple-evals release](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py). The two final checkpoint IDs and SHA-256 hashes are recorded in the numerical summary. They are available in the HPG confirmation archive. The [registered confirmation recipe](../codex/KAM_PHASE6_CONFIRMATION_V2_PREREGISTRATION.md) specifies the training corpus and run settings.

1. Stage the [GPU runner](../../scripts/run_kam_general_benchmarks.py), [code grader](../../scripts/grade_kam_general_code.py), and Slurm files into an HPG run directory. Place the [Phase VI model source](https://github.com/Jibby2k1/KAM/tree/fa888b99fffb9413222f0f64cfee931d0f95072e) under its repo path and benchmark assets under its data path. Install PyArrow 20.0.0 into its runtime path under the PyTorch 2.8.0 module. The second argument of the GPU job is the archive directory containing the manifest and checkpoints.
2. Run the [GPU array](../../slurm/kam_general_benchmarks.sbatch). The faster replication path omits indices 0, 2 and 8 and runs the [KAM generation array](../../slurm/kam_general_generation_exact.sbatch) for those tasks. It skips unused statistics after checking exact logits and greedy prefixes against the archived code on the same GPU. The alternative reference path uses the [unmodified generation shards](../../slurm/kam_general_generation_shards.sbatch) and [merge job](../../slurm/kam_general_generation_merge.sbatch). After the faster path is checked against any completed reference output, copy the three fast results to their matching unsuffixed file names for scoring. Stage a Python 3.11.14 slim SIF as python.sif and submit the [code grading array](../../slurm/kam_general_code_grading.sbatch) after successful generation.
3. Run the [summarizer](../../scripts/summarize_kam_general_benchmarks.py) on the resulting directory or the published per-item gzip file. It verifies source, checkpoint, and per-question prompt length agreement. Its paired item bootstrap intervals describe this checkpoint pair. They do not capture variation across training seeds.
