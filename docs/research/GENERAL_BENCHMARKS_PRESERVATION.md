# Preserving pretrained benchmark scores during KAM adapter training

This study uses [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) from Hugging Face. Untouched SmolLM2 is the teacher and the evaluation reference. UF Research Computing prohibits Qwen use on HiPerGator. The previous Qwen result is a historical record. This study used no Qwen checkpoint, adapter, teacher outputs, or tokenized corpus. [UF Research Computing policy](https://docs.rc.ufl.edu/training/new_user_training/#prohibition-on-large-language-models-from-countries-of-concern)

## HPG results

Adapter cells give the mean across three seeds. Within each cell, KAM appears first and dot second.

| Benchmark | Untouched | Reference KAM / dot | Lower LR KAM / dot | Lower LR + KL KAM / dot |
| --- | ---: | ---: | ---: | ---: |
| HumanEval | 28.66% | 30.89% / 29.07% | 28.05% / 27.03% | 27.64% / 28.46% |
| MBPP | 38.00% | 36.87% / 32.33% | 37.20% / 33.67% | 38.13% / 37.07% |
| GPQA Diamond | 25.76% | 24.92% / 24.41% | 26.77% / 23.74% | 26.26% / 23.06% |
| MMLU | 42.50% | 38.45% / 39.16% | 42.50% / 36.51% | 42.54% / 42.09% |
| GSM8K | 31.77% | 12.31% / 6.92% | 30.00% / 3.59% | 30.50% / 27.29% |

| Benchmark | Lower LR − reference, KAM / dot (pp) | KL − lower LR, KAM / dot (pp) | Lower LR + KL − untouched, KAM / dot (pp) |
| --- | ---: | ---: | ---: |
| HumanEval | -2.85 / -2.03 | -0.41 / +1.42 | -1.02 / -0.20 |
| MBPP | +0.33 / +1.33 | +0.93 / +3.40 | +0.13 / -0.93 |
| GPQA Diamond | +1.85 / -0.67 | -0.51 / -0.67 | +0.51 / -2.69 |
| MMLU | +4.04 / -2.64 | +0.05 / +5.58 | +0.04 / -0.41 |
| GSM8K | +17.69 / -3.34 | +0.51 / +23.70 | -1.26 / -4.47 |

| Recipe | Validation KL, KAM / dot (nats/token) | Validation mixture NLL, KAM / dot (nats/token) |
| --- | ---: | ---: |
| Reference | 0.14263 / 0.76966 | 1.7414 / 1.5652 |
| Lower LR | 0.00493 / 0.39922 | 1.9170 / 1.6534 |
| Lower LR + KL | 0.00263 / 0.01176 | 1.9234 / 1.7918 |

Lowering the learning rate recovered 4.04 points on MMLU and 17.69 on GSM8K for KAM. Adding KL at that rate changed those scores by +0.05 and +0.51 points, with both paired intervals spanning zero. The dot adapter gained 5.58 MMLU points (95% item interval +4.93 to +6.21) and 23.70 GSM8K points (+21.38 to +26.06) from adding KL at the lower rate. Against untouched SmolLM2, the combined recipe still lost 1.26 GSM8K points for KAM (−2.38 to −0.18) and 4.47 for dot (−6.54 to −2.40).

The per-seed score differences and the conditional three-seed item intervals are in the [numerical summary](../../results/general_benchmarks_preservation/summary.json). The [per-item archive](../../results/general_benchmarks_preservation/paired-outcomes.json.gz) contains numerical outcomes and training manifests. It contains no generated code or dataset prose.

### Per-seed counts

Seed order is 731201, 731211, 731221.

| Benchmark | Recipe | KAM correct across three seeds | Dot correct across three seeds |
| --- | --- | ---: | ---: |
| HumanEval (164) | Reference | 48, 52, 52 | 48, 48, 47 |
| HumanEval (164) | Lower LR | 45, 47, 46 | 45, 43, 45 |
| HumanEval (164) | Lower LR + KL | 45, 47, 44 | 47, 46, 47 |
| MBPP (500) | Reference | 182, 186, 185 | 168, 161, 156 |
| MBPP (500) | Lower LR | 186, 190, 182 | 171, 168, 166 |
| MBPP (500) | Lower LR + KL | 189, 187, 196 | 186, 191, 179 |
| GPQA Diamond (198) | Reference | 45, 52, 51 | 51, 44, 50 |
| GPQA Diamond (198) | Lower LR | 54, 52, 53 | 46, 49, 46 |
| GPQA Diamond (198) | Lower LR + KL | 54, 51, 51 | 43, 46, 48 |
| MMLU (14,042) | Reference | 5,546, 5,226, 5,427 | 5,428, 5,380, 5,687 |
| MMLU (14,042) | Lower LR | 5,973, 5,969, 5,960 | 5,111, 5,087, 5,184 |
| MMLU (14,042) | Lower LR + KL | 5,975, 5,970, 5,976 | 5,882, 5,959, 5,890 |
| GSM8K (1,319) | Reference | 145, 191, 151 | 85, 59, 130 |
| GSM8K (1,319) | Lower LR | 399, 389, 399 | 40, 50, 52 |
| GSM8K (1,319) | Lower LR + KL | 402, 400, 405 | 353, 369, 358 |

## Design

For seeds 731201, 731211, and 731221, both arms begin from the same frozen backbone and identical projections at four equally spaced decoder layers. The output projections start at zero. Each KAM branch uses exact top-four routing over 256 fixed keys with learned values. Each control branch uses causal dot attention over input tokens. Each adapter stack has 1,114,112 trainable parameters. The fixed KAM key buffers are additional state.

| Recipe | Peak rate | Forward KL weight |
| --- | ---: | ---: |
| Reference | 0.0008 | 0 |
| Lower rate | 0.00008 | 0 |
| Lower rate + KL | 0.00008 | 4 |

Each fit sees 1,048,576 subword tokens over 256 updates. A batch contains eight windows of 512 tokens with a 4:2:2 mix of WikiText prose, Python code, and independently generated arithmetic from the [frozen source mix](GENERAL_BENCHMARKS_LONG.md). The three seed-specific window lists are identical across architectures and recipes. AdamW warms up over 16 steps and then decays by cosine to one tenth of the peak. The gradient norm cap is one.

The KL term is KL(untouched teacher ∥ adapted student) at temperature one over the full vocabulary. It is averaged over one quarter of positions per batch, with the offset rotating by update. Cross-entropy uses all positions. The same frozen backbone computes both distributions. The teacher forward bypasses every adapter and runs without gradients. Validation reports domain NLL and the average token KL under a fixed recorded schedule. No benchmark item selects a checkpoint. The recipe set estimates the lower-rate effect and the incremental KL effect at that rate. It does not estimate a rate-by-KL interaction.

## Evaluation

All five benchmark files and scoring metrics match the preceding studies. The same zero-shot text prompts and SmolLM2 chat template are used across the 19 checkpoints. HumanEval and MBPP generate complete programs. MBPP includes the first public example assertion. Extracted code blocks have common indentation removed uniformly and are graded in the same no-network container. GPQA and MMLU score four option letters, and GSM8K uses strict marked numeric exact match. The generation limits are 256 new subword tokens for programming and 384 for GSM8K. Prompts may contain up to 4,096 tokens. Cache calibration compares cached and full readouts on two prompts per task and checkpoint.

The headline comparisons are seed means. Each pair shares the benchmark item IDs and the same three seeds. The item bootstrap resamples items jointly across those three fixed seeds. Its intervals are conditional on those fits, and the report gives each seed's score to show training variation. They do not adjust for multiple comparisons. “Preserve” refers to the observed gap from the untouched checkpoint. No noninferiority margin was specified.

HPG job 53045070 retokenized the source mix. Array 53045071 trained all 18 adapters. Seeds 731201 and 731211 ran on B200s. All six fits for seed 731221 ran on RTX PRO 6000s. The benchmark arrays 53045344 and 53045345 used RTX PRO 6000s. CPU grading arrays were 53045346 and 53045347. See the [model fingerprints](../../results/general_benchmarks_preservation/model-sha256.txt) and [tokenized corpus manifest](../../results/general_benchmarks_preservation/token-manifest.json).

## Reproduce

1. Place the pinned model in model, the KAM package in repo, the previous untokenized source mixture in corpus, the benchmark inputs in data and the PyArrow runtime in runtime under the HPG run directory. Use PyTorch 2.8.0, Transformers 5.14.0 and the isolated Python image from the [benchmark protocol](GENERAL_BENCHMARKS_PROTOCOL.md).
2. Record the model identifier and revision on two lines in model-origin.txt, then SHA-256 hashes of config.json, tokenizer.json and model.safetensors in model-sha256.txt. Use the [retokenization array](../../slurm/kam_pretrained_prepare.sbatch) and the [matched training array](../../slurm/kam_adapter_preservation_train.sbatch).
3. Run the [evaluation array](../../slurm/kam_adapter_preservation_eval.sbatch) and the [isolated coding grades](../../slurm/kam_adapter_preservation_grade.sbatch). Use the [summarizer](../../scripts/summarize_kam_adapter_preservation.py) on the per-checkpoint numeric files or their published gzip archive. The runner is pinned to the model registry and does not load the historical Qwen checkpoint.
