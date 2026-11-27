# Development-selected KL on fresh math problems

This experiment reuses the frozen SmolLM2-1.7B-Instruct and UF-hosted Llama 3 8B Instruct checkpoints from the [matched factorial](GENERAL_BENCHMARKS_FACTORIAL.md) and [intermediate-KL follow-up](GENERAL_BENCHMARKS_DIAGNOSTICS.md#intermediate-kl-at-the-lower-learning-rate). The learning rate is fixed at 0.00008. KAM and dot use matched trainable parameter counts, training exposure, and training hardware within seed. The three seeds are 731201, 731211, and 731221. All new evaluations use HPG RTX PRO 6000 GPUs.

## Selection and evaluation

The development set contains 512 unique GSM8K problems from the authors' original training split. It is held out from the adapter training streams. We selected the lowest SHA-256 ranks of the eligible source row indices under the registered protocol string. Development excludes duplicates and exact normalized matches to the previously evaluated GSM8K test and SVAMP prompts. [Pinned GSM8K training split](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl)

For each backbone, the same weights 0, 0.5, and 4 are eligible for both architectures. Each candidate is ranked by the number of development problems answered exactly under **both** formats, pooled across both architectures and all three seeds. Ties choose the smaller weight. The winner is shared by KAM and dot within that backbone. The new evaluation uses only that weight and the untouched backbone.

The fresh test fixture contains all 311 of the released MATH-500 problems whose authored answer is a bare signed integer. These span all seven subjects and five difficulty levels. The two fixtures were locked together before development decoding. We excluded exact normalized duplicates and matches to the full GSM8K train and test splits and SVAMP. All 311 integer-eligible test items remained. This is **MATH-500 integer-subset exact match**, with the same test denominator for every model and format. [Pinned MATH-500 release](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl) · [Original evaluation split](https://github.com/openai/prm800k#math-splits)

## Answer formats

Each prompt asks for the final number on a line beginning with either `#### ` or `Answer: `. The wording differs only in the marker. Both runs use greedy generation with a 384-token cap and the same batch membership and order within a backbone. Generation uses the same numerical stop rule instantiated with the requested marker. The parser accepts a standalone marked decimal number, including a leading minus sign, valid comma grouping, and optional final punctuation. Exact decimal comparison credits numerically equivalent decimal spellings. The two primary format scores are strict exact match. The both-format score requires the same item to be correct in each separate run.

The exact normalized prompt audit found no matches in the frozen Wikipedia, Python, or arithmetic adapter-training streams for either backbone. Public development and test datasets can appear in backbone pretraining or related corpora. [Source overlap audit](../../results/general_benchmarks_math_kl/source-overlap.json)

## Development results

Cells count problems correct under both markers across 512 items, two architectures, and three seeds, for 3,072 paired attempts per cell.

| Backbone | KL 0 | KL 0.5 | KL 4 | Selected weight |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 | 131 (4.26%) | 142 (4.62%) | **224 (7.29%)** | **4** |
| Llama 3 | 1,620 (52.73%) | **1,891 (61.56%)** | 1,857 (60.45%) | **0.5** |

The Llama winner exceeded weight 4 by 34 paired attempts (1.11 percentage points). The SmolLM2 winner exceeded weight 0.5 by 82 paired attempts (2.67 points). These are development selection scores. [SmolLM2 selection and dev breakdown](../../results/general_benchmarks_math_kl/smollm2-selection.json) · [Llama selection and dev breakdown](../../results/general_benchmarks_math_kl/llama3-selection.json)

## Fresh MATH-500 integer-subset results

Adapter scores are means across the three matched seeds. Differences are in percentage points, with 95% paired item bootstrap intervals conditional on these seeds.

| Backbone | Format | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2, KL 4 | `####` | 12.22% | 13.72% | 9.43% | +4.29 [+2.04, +6.75] |
| SmolLM2, KL 4 | `Answer:` | 8.68% | 9.43% | 9.32% | +0.11 [−2.14, +2.36] |
| SmolLM2, KL 4 | Both | 4.18% | 4.82% | 4.50% | +0.32 [−0.96, +1.71] |
| Llama 3, KL 0.5 | `####` | 24.44% | 23.26% | 27.33% | −4.07 [−7.61, −0.54] |
| Llama 3, KL 0.5 | `Answer:` | 29.26% | 26.26% | 26.26% | 0.00 [−3.32, +3.32] |
| Llama 3, KL 0.5 | Both | 19.29% | 18.11% | 21.97% | −3.86 [−7.18, −0.64] |

SmolLM2 KAM led under `####` in every seed. The models finished within 0.11 points under `Answer:` and within 0.32 points on the both-format metric. Llama dot led on both-format correctness in all three seeds, by 4.18, 3.22, and 4.18 points. Under `Answer:`, the two Llama adapters tied in aggregate, and the untouched model scored 29.26%.

On the Llama both-format metric, KAM gained a net 2.0 correct items per seed on the subset where both adapters emitted both requested markers. On the one-adapter-only subsets, KAM gained 8.7 and dot gained 22.7 correct items per seed. That accounts for dot's aggregate 12-item lead per seed. On SmolLM2's `####` run, KAM gained 13.0 net items per seed where both adapters emitted a mark, out of its total gain of 13.3. These partitions describe observed completions. [SmolLM2 per-seed scores, contrasts and parse counts](../../results/general_benchmarks_math_kl/smollm2-test-summary.json) · [Llama per-seed scores, contrasts and parse counts](../../results/general_benchmarks_math_kl/llama3-test-summary.json)

The fresh arrays were submitted after all 36 development runs finished and used the digests of the two frozen selections. [SmolLM2 development receipt](../../results/general_benchmarks_math_kl/smollm2-dev-scheduler.tsv) · [Llama development receipt](../../results/general_benchmarks_math_kl/llama3-dev-scheduler.tsv) · [SmolLM2 test receipt](../../results/general_benchmarks_math_kl/smollm2-test-scheduler.tsv) · [Llama test receipt](../../results/general_benchmarks_math_kl/llama3-test-scheduler.tsv)

## Reproduction

Rebuild the two fixtures with the protocol script's `prepare` command and verify the [fixture digests and IDs](../../results/general_benchmarks_math_kl/fixture-manifest.json). Stage the shared `eval_kam_pretrained.py` beside the experiment evaluator and run the `dev` tasks 0–17 on each prepared checkpoint root. Use the summarizer to produce `selection.json` with the original factorial and intermediate-KL archives as checkpoint references. Run `test` tasks 0–6 with that selection file, then run the summarizer with `--test-output`. The test stage records the selection digest, and the summarizer recomputes the selection before accepting test records. To replay from the compressed per-checkpoint outcomes, expand each archive key to its own JSON file with sorted keys, compact separators and a final newline, matching the evaluator's byte format. [SmolLM2 outcomes](../../results/general_benchmarks_math_kl/smollm2-outcomes.json.gz) · [Llama outcomes](../../results/general_benchmarks_math_kl/llama3-outcomes.json.gz) · [Fixture and scoring protocol](../../scripts/kam_math_kl_protocol.py) · [GPU evaluator](../../scripts/eval_kam_math_kl_selection.py) · [Slurm driver](../../slurm/kam_math_kl_selection.sbatch) · [Selection and summary](../../scripts/summarize_kam_math_kl_selection.py)
