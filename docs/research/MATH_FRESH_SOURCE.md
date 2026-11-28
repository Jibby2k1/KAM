# An independent source for quantitative reasoning

The 478-item [balanced replication](MATH_ANSWER_TYPES.md) used different questions from the same original MATH family. The next question is whether a comparison on a separate source can support a useful paired estimate. Omni-MATH supplies 4,428 annotated competition questions across 67 source labels. Its release date alone cannot establish when an individual problem first became public. [Omni-MATH paper](https://arxiv.org/abs/2410.07985) · [Pinned data release](https://github.com/KbsdJames/Omni-MATH/blob/5aac08bc8ebeef5511e244c094194321209550af/Omni-Math.jsonl)

## Input and answer audit

Normalizing Unicode, case and punctuation found 16 exact prompt overlaps with original MATH training rows, six with original MATH test rows outside MATH-500, and zero with MATH-500, either GSM8K source split, or SVAMP. Sixty-eight rows belonged to duplicate prompt groups within Omni-MATH. Excluding repeats and all known overlaps left 4,338 unique rows. The audit covers named inputs and exact normalized duplicates. A full study should also screen paraphrases and separate model-exposure provenance from the archive's own release date. [Audit source](../../scripts/audit_kam_omni_math_fixture.py)

| Difficulty in Omni-MATH | Plain scalar answer | Other self-verifying extraction | Unverified or empty |
| --- | ---: | ---: | ---: |
| At most 2 | 326 | 121 | 5 |
| Above 2 through 4 | 552 | 354 | 18 |
| Above 4 through 6 | 922 | 913 | 102 |
| Above 6 | 139 | 636 | 250 |
| Total | 1,939 | 2,024 | 375 |

A plain scalar is a signed or unsigned integer or decimal with a successful gold self-check. The other column counts successful first-match extraction and self-equivalence. Complete proof, text, and multipart answer coverage would need a separate annotation audit. The 1,939 scalar rows make a large matched natural cohort possible if the untouched models retain enough accuracy across difficulty bands. The easiest two bands together supply 878 scalar items. [Numeric inventory](../../results/general_benchmarks_omni_math/feasibility.json)

## Locked untouched-backbone pilot

Use the 12 hash-first unseen scalar prompts in each of the four difficulty bands, excluding them from any later evaluation. Measure the untouched SmolLM2 and Llama backbones with the same neutral prompt, 1,024-token maximum, greedy native chat templates and shared grader on HPG RTX PRO 6000. Record counts of correct, parsed and capped responses by band. This small check measures feasibility and informs the priority of a larger study. The full remaining pool has 1,891 candidate scalar prompts before near-duplicate or coverage review. Weight future precision calculations by difficulty and source, and preserve the development-selected KL weights for the fitted arms. [Locked pilot manifest](../../results/general_benchmarks_omni_math/untouched-pilot-manifest.json) · [Pilot evaluator](../../scripts/kam_omni_math_untouched_pilot.py)

| Difficulty band | SmolLM2 correct / 12 | Llama correct / 12 | SmolLM2 / Llama cap hits |
| --- | ---: | ---: | ---: |
| At most 2 | 5 | 5 | 1 / 0 |
| Above 2 through 4 | 1 | 1 | 1 / 3 |
| Above 4 through 6 | 0 | 1 | 5 / 3 |
| Above 6 | 0 | 0 | 2 / 3 |

The same two backbones scored five of 12 in the easiest band, then one and two correct, respectively, across the 36 harder prompts. SmolLM2 had 48 parseable answers and Llama had 46, with zero grader errors. The small stratified pilot suggests that the full competition mix would spend much of the compute at a score floor. For a primary paired comparison, prioritize independent sources with more solvable items and retain higher difficulty as a labeled stress slice. The four strata were equally represented by design. [SmolLM2 results](../../results/general_benchmarks_omni_math/smollm2-untouched-pilot.json) · [Llama results](../../results/general_benchmarks_omni_math/llama3-untouched-pilot.json) · [HPG receipt](../../results/general_benchmarks_omni_math/untouched-pilot-scheduler.tsv)
