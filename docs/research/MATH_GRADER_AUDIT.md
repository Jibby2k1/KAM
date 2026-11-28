# Stress-testing the shared math grader

The [masked retention audit](MATH_ANSWER_TYPES.md#retention-on-the-balanced-cohort) found three Llama examples with an equivalent written answer that the shared first-match grader missed. This controlled follow-up separates extraction from the mathematical comparison. Thirteen authored cases cover integers, fractions, percentages, radicals, complex numbers, functions, intervals, sets, ordered pairs, and inequalities. Each case has one correct and one plausible incorrect final answer. The frozen symbolic verifier distinguished these pairs when the expressions were supplied directly.

Render each pair in six short response forms. In the two decoy forms, put the opposite answer in an earlier sentence as displayed math. Keep the final claim clearly identified in prose in both versions. This tests whether an early, mathematical-looking expression displaces the actual conclusion. [Exact authored cases and renderers](../../scripts/audit_kam_math_grader_surface_forms.py)

| Rendering | Equivalent finals credited, of 13 | Inequivalent finals credited, of 13 |
| --- | ---: | ---: |
| Bare expression | 2 | 0 |
| Bare after an optional final-answer cue | 2 | 0 |
| Bare after a delimited decoy | 2 | 8 |
| Delimited math | 13 | 0 |
| Boxed math | 13 | 0 |
| Delimited math after a delimited decoy | 13 | 0 |

The same results held when scoring the last three nonempty lines because each authored response fits inside three lines. In a separate comparator check, the pinned grader also rejected the equivalence of `t ≥ −11` and `−11 ≤ t` when both were supplied directly. These are constructed stress cases and give no estimate of extraction error prevalence on the benchmark. The official results retain the same shared grader and marker-free prompt. [Numeric replay](../../results/general_benchmarks_full_math500/surface-form-audit.json)

An extraction improvement should pass both sides of this test. It needs to recover final expressions written in prose and reject an early correct expression when the response later gives a wrong answer. A held-out audit should stratify by answer type and compare with masked judgments on actual model generations before changing a benchmark protocol.
