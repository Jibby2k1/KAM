# KAM adapters on a shared pretrained Qwen backbone

The three arms start from the same [Qwen2.5-1.5B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306). The untouched arm measures the quality available at initialization. The other two freeze all backbone weights, including the language-model head, and train residual adapters at decoder layers 8, 15, 22, and 28, counted from one.

## HPG results

| Benchmark | Untouched Qwen | KAM adapter | Dot adapter | KAM − dot (pp) | Paired 95% item interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 50.61% (83/164) | 47.56% (78/164) | 42.07% (69/164) | +5.49 | [-0.61, +11.59] |
| MBPP | 48.20% (241/500) | 33.40% (167/500) | 41.80% (209/500) | -8.40 | [-12.60, -4.40] |
| GPQA Diamond | 30.30% (60/198) | 23.74% (47/198) | 29.29% (58/198) | -5.56 | [-10.61, -0.51] |
| MMLU | 57.75% (8,109/14,042) | 54.45% (7,646/14,042) | 55.34% (7,771/14,042) | -0.89 | [-1.47, -0.31] |
| GSM8K | 21.99% (290/1,319) | 17.29% (228/1,319) | 16.45% (217/1,319) | +0.83 | [-1.59, +3.26] |

- HumanEval graded outcomes. Untouched 49 failed tests, 83 passed, 32 syntax error. KAM 60 failed tests, 78 passed, 26 syntax error. Dot 71 failed tests, 69 passed, 24 syntax error.
- Before common indentation was removed, raw MBPP pass counts were 241 for untouched Qwen, 0 for KAM, and 209 for dot.
- MBPP graded outcomes. Untouched 248 failed tests, 241 passed, 11 syntax error. KAM 190 failed tests, 167 passed, 143 syntax error. Dot 260 failed tests, 9 no definition, 209 passed, 22 syntax error.
- On GSM8K, parseable marked answers were 651/1,319 for untouched Qwen, 1,191 for KAM and 1,283 for dot.

| Matched held-out next-token NLL (nats per subword) | Untouched Qwen | KAM adapter | Dot adapter |
| --- | ---: | ---: | ---: |
| Wiki | 2.3481 | 2.1573 | 2.1112 |
| Code | 1.5165 | 1.4395 | 1.4042 |
| Math | 1.0608 | 0.3800 | 0.3277 |
| B200 elapsed training and scheduled validation (s) | — | 27.4 | 23.4 |

Untouched Qwen had the highest observed score on all five benchmarks. Against the matched dot arm, KAM scored lower on MBPP, GPQA, and MMLU. Its point estimates were higher on HumanEval and GSM8K, with paired intervals spanning zero. Both trained adapters lowered held-out loss on the mixture while lowering the five benchmark point estimates. The item intervals are unadjusted and describe one trained pair. The [numerical summary](../../results/general_benchmarks_pretrained/summary.json) also reports paired changes from untouched Qwen, prompt lengths, letter counts, and cached versus full calibration. The [per-item numerical archive](../../results/general_benchmarks_pretrained/paired-outcomes.json.gz) contains no generated code or dataset prose.

## Matched intervention

Both adapters have an RMS-normalized input, a 1,536-to-64 projection, and a 64-to-1,536 output projection. Their common projections start from identical weights across arms, including an all-zero output projection. Each adapter starts with the same tested logits as the untouched backbone. Each trained arm adds **851,968 trainable parameters**.

| Adapter | Operation in the 64-dimensional branch | Added parameters in that operation per layer |
| --- | --- | ---: |
| KAM | Exact four-nearest support routing by dot score over 256 fixed unit keys, then a weighted sum of trainable value vectors | 16,384 |
| Dot attention | Four-head causal attention over the current input sequence with trainable Q, K, V, and output projections | 16,384 |

KAM stores 65,536 additional fixed key scalars across its four layers. The arms match trained parameters and training exposure. Their operation counts and memory access differ. The pretrained causal attention blocks remain active in every arm.

## Training

The two adapters train with next-token loss and the same 512 AdamW updates, 512-token windows, and 2,097,152 sampled subword tokens. Update batches contain four prose, two Python, and two independently generated arithmetic windows from the [previously frozen mixture](GENERAL_BENCHMARKS_LONG.md). The packed sources are retokenized with the pretrained tokenizer. Optimization uses BF16 autocast, a warmup of 32 updates, a peak rate of 0.0008, cosine decay to one-tenth of the peak rate, and gradient clipping at one. This is one optimization run per architecture. It measures the effect of this fixed adapter recipe.

## Benchmarks

The three arms use the same benchmark files and scoring metrics. All initial prompts fit in 4,096 native tokens. The Qwen chat template wraps the prompts equally across arms. This recipe differs from the earlier byte model evaluation.

- **GPQA Diamond and MMLU** use zero-shot choice letter logits after the assistant cue. GPQA keeps the previous deterministic per-question option order. MMLU reports micro-averaged accuracy across 57 subjects.
- **HumanEval** asks for the full Python program. **MBPP** asks for a full function and includes the first public example assertion to specify its interface. The grader uses the same benchmark unit tests and reports pass@1. Coding generates up to 256 subword tokens and extracts the first fenced code block when present. The reported scores remove common block indentation uniformly across all three arms. This normalization was added after all 500 raw KAM MBPP outputs failed Python parsing with an unexpected initial indent. The [raw coding results](../../results/general_benchmarks_pretrained/raw-code-outcomes.json.gz) preserve the original grading behavior.
- **GSM8K** generates up to 384 subword tokens and scores the same strict marked final-answer exact match. An unparseable answer is counted wrong.
- Generation uses greedy decoding and cached attention. A calibration compares cached and full readouts. The isolated code grader keeps generated programs and dataset text in cluster storage.

Benchmark overlap in the upstream pretrained data is unknown. The matched test measures changes relative to the same pinned pretrained checkpoint. It does not estimate performance on a fresh, contamination-controlled benchmark.

HPG job 53038499 tokenized the corpus. B200 array 53038500 trained both adapters. Evaluation arrays 53038808 and 53038810 used RTX PRO 6000 GPUs. Isolated coding jobs 53038811 and 53038814 retained raw grading outcomes on HPG CPUs. Array 53040218 regraded the same completions with common block indentation removed. The HPG runtime used PyTorch 2.8.0 and Transformers 5.14.0.

## Reproduce

1. Fetch the [pinned benchmark inputs](../../scripts/fetch_kam_general_benchmarks.sh), [parent corpus](GENERAL_BENCHMARKS_LONG.md), and the pinned model files from the revision above. Stage the [model adapters](../../scripts/kam_pretrained_adapters.py), Python entry points, and Slurm files in the HPG run directory. Use PyTorch 2.8.0 with Transformers 5.14.0. Put the model under the run directory's model path, the KAM package under repo, the frozen previous corpus under corpus, and the benchmark environment from the previous study under runtime.
2. Run the [tokenization job](../../slurm/kam_pretrained_prepare.sbatch), then the [B200 adapter training array](../../slurm/kam_pretrained_train.sbatch). Retain the adapter weights from the two arms and their [KAM training record](../../results/general_benchmarks_pretrained/train-kam.json) and [dot training record](../../results/general_benchmarks_pretrained/train-dot.json). The [token manifest](../../results/general_benchmarks_pretrained/token-manifest.json) identifies the retokenized inputs. Record the SHA-256 digests of config.json, tokenizer.json, and model.safetensors in model-sha256.txt from the model directory.
3. Run the [RTX PRO 6000 benchmark array](../../slurm/kam_pretrained_eval.sbatch) and the [CPU coding grades](../../slurm/kam_pretrained_grade.sbatch). The same [summarizer](../../scripts/summarize_kam_pretrained.py) reads either the per-arm numeric outputs or the published gzip. It checks checkpoints, exposure, prompt lengths, dataset hashes, and coverage.
