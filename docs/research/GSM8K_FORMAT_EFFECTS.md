# How answer requirements change the measured KAM–dot gap

The full [neutral-prompt MATH-500 comparison](FULL_MATH500.md) found small KAM–dot gaps. An earlier [two-marker comparison](MATH_KL_TRANSFER.md) had larger format-dependent gaps. We tested the prompt request and the grading rule as separate interventions on 512 GSM8K questions that were unused in previous KAM evaluations and development selection.

The pinned questions are hash-ranked from the authors' source-training split after excluding earlier development questions, duplicate prompts, and exact matches to previous GSM8K test, SVAMP, and MATH-500 prompts. The exact normalized overlap audit also found zero matches against adapter-training text. Both backbones retain their development-selected KL weights, with KAM and dot matched on architecture parameter count, training hardware within seed, and three seeds. Each includes the untouched checkpoint. [Pinned GSM8K source](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl) · [Fixture manifest](../../results/general_benchmarks_gsm8k_formats/fixture-manifest.json) · [Training source audit](../../results/general_benchmarks_gsm8k_formats/source-overlap.json)

## What changes

1. **Prompt request.** The neutral prompt asks the model to state the final answer clearly. The two marked prompts add one sentence asking for `####` or `Answer:` on a new line. Every run uses native chat templates, greedy decoding, the same batch membership, and stops at EOS or the 384-token cap. All three outputs are first graded with the shared Math-Verify protocol.
2. **Grading rule.** Regrade the exact same marked outputs using the locked strict numeric parser. It requires a standalone marker and a bare numeric answer with optional punctuation. Unit words and currency symbols can make an otherwise recognizable correct answer fail this syntax. Every item stays in the denominator.

## Scores

The cells are exact answer scores, averaged across the same three seeds. The paired intervals compare KAM with dot across questions conditional on those seeds.

| Backbone | Prompt | Grader | Untouched | KAM | Dot | KAM minus dot (95% interval) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2 | Neutral | Shared | 49.61% | 50.72% | 50.72% | 0.00 [−3.19, +3.06] pp |
| SmolLM2 | `####` | Shared | 48.05% | 47.46% | 43.36% | +4.10 [+0.26, +8.01] pp |
| SmolLM2 | `####` | Strict | 13.28% | 14.19% | 11.07% | +3.12 [+0.33, +5.92] pp |
| SmolLM2 | `Answer:` | Shared | 32.23% | 31.84% | 25.78% | +6.05 [+3.19, +8.92] pp |
| SmolLM2 | `Answer:` | Strict | 0.98% | 1.11% | 0.65% | +0.46 [−0.39, +1.37] pp |
| Llama 3 | Neutral | Shared | 83.40% | 80.27% | 79.23% | +1.04 [−1.37, +3.45] pp |
| Llama 3 | `####` | Shared | 85.16% | 87.37% | 84.64% | +2.73 [+0.65, +4.82] pp |
| Llama 3 | `####` | Strict | 45.70% | 53.45% | 58.92% | −5.47 [−8.27, −2.73] pp |
| Llama 3 | `Answer:` | Shared | 85.16% | 86.59% | 84.11% | +2.47 [+0.33, +4.69] pp |
| Llama 3 | `Answer:` | Strict | 43.16% | 46.48% | 51.89% | −5.40 [−8.07, −2.80] pp |

## Separate prompt effects from grading effects

| Backbone | Marker | Change in KAM–dot gap from prompt request | Change from strict grading on the same outputs |
| --- | --- | ---: | ---: |
| SmolLM2 | `####` | +4.10 [−0.72, +8.92] pp | −0.98 [−4.82, +2.86] pp |
| SmolLM2 | `Answer:` | +6.05 [+1.63, +10.55] pp | −5.60 [−8.53, −2.67] pp |
| Llama 3 | `####` | +1.69 [−1.04, +4.36] pp | −8.20 [−10.61, −5.79] pp |
| Llama 3 | `Answer:` | +1.43 [−1.17, +4.04] pp | −7.88 [−10.35, −5.47] pp |

For Llama, changing the grader alone reversed the KAM–dot ordering with either marker. Across both fitted arms and three seeds, the strict parser rejected all 919 hash-prompt completions that were scored correct only by the shared grader. A literal hash marker still appeared in 892 of those outputs. For the answer prompt the corresponding counts were 1,111 rejected, with the literal marker in 1,083. Fixed hash-selected examples include a marked dollar amount and a marked number followed by units. The strict grammar counts these as unparsed. The presence count records any literal occurrence in the completion, including an occurrence outside the final line.

The prompt request itself also mattered. On SmolLM2, adding `Answer:` reduced shared-grader accuracy by 18.88 points for KAM and 24.93 for dot. On Llama, both marker requests improved shared-grader accuracy on the tested prompts. The neutral prompt remains a separate evaluation condition. [SmolLM2 numeric summary](../../results/general_benchmarks_gsm8k_formats/smollm2-summary.json) · [Llama numeric summary](../../results/general_benchmarks_gsm8k_formats/llama3-summary.json)

### Require a marker and allow equivalent expressions

An additional audit isolates the bare-number grammar. It takes the last line that begins with the requested marker and passes everything after the marker to the shared grader. This still requires a marker at the start of an answer line and allows mathematically appropriate notation such as currency or units in the answer.

| Llama prompt | Shared on entire output KAM / dot | Shared after marked line KAM / dot | Strict marker and bare number KAM / dot |
| --- | ---: | ---: | ---: |
| `####` | 87.37% / 84.64% | 86.20% / 84.18% | 53.45% / 58.92% |
| `Answer:` | 86.59% / 84.11% | 85.42% / 83.20% | 46.48% / 51.89% |

Requiring a marked line reduced the KAM–dot gap by 0.72 points for the hash prompt (95% paired item interval −1.69 to +0.20) and 0.26 for the answer prompt (−1.24 to +0.72). The strict bare-number parser shifted the gap a further 7.49 and 7.62 points toward dot, respectively. Both fitted arms used the same generated outputs throughout this audit. [SmolLM2 detail](../../results/general_benchmarks_gsm8k_formats/smollm2-marked-equivalence.json) · [Llama detail](../../results/general_benchmarks_gsm8k_formats/llama3-marked-equivalence.json) · [Audit code](../../scripts/analyze_kam_gsm8k_marked_equivalence.py)

## Reproduce

Use the [fixture prep, evaluator and grader](../../scripts/kam_gsm8k_format_effects.py) with the pinned source files and previous [selection manifest](../../results/general_benchmarks_math_kl/fixture-manifest.json). The [Slurm driver](../../slurm/kam_gsm8k_format_effects.sbatch) runs each checkpoint once for all three prompts using the shared generation routine. The [summarizer](../../scripts/summarize_kam_gsm8k_format_effects.py) verifies checkpoint and fixture identity and checks that the prompt and grading changes sum to the total change. [SmolLM2 outcomes](../../results/general_benchmarks_gsm8k_formats/smollm2-outcomes.json.gz) · [Llama outcomes](../../results/general_benchmarks_gsm8k_formats/llama3-outcomes.json.gz) · [SmolLM2 receipt](../../results/general_benchmarks_gsm8k_formats/smollm2-scheduler.tsv) · [Llama receipt](../../results/general_benchmarks_gsm8k_formats/llama3-scheduler.tsv)
