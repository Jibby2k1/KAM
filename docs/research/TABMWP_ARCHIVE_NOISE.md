# Tabular arithmetic with an archived distractor

This pilot tests whether fixed-KL KAM and dot adapters keep answering from a labeled current table when another plausible table is present. TabMWP supplies naturally authored tables and questions at multiple grade levels. The distractor table comes from a distinct development example with the same topic title, normalized header, grade, and row count. The current and archive roles are assigned for the experiment. [TabMWP paper](https://openreview.net/forum?id=DHyHRBwJUTN) · [Pinned data](https://github.com/lupantech/PromptPG/tree/869bae8ffb8288ccc3102293b3090ec2aa17f618/data/tabmwp)

## Locked fixture and measurements

Use 128 TabMWP test questions, 32 from each of grades five through eight. Questions are free-text with plain scalar references. Require the shared symbolic gold self-check and exclude exact table duplicates across used target and donor tables. The inventory of test rows eligible for a matched donor has 407, 238, 501, and 486 rows across the four grades. The frozen sample has 92 title-and-header groups. It is a controlled sample of the test release. [Fixture manifest](../../results/general_benchmarks_tabmwp/fixture-manifest.json)

Each question is presented three times with the same current table, question and instructions. With no archive, the prompt contains only the current table. In the other two prompts, the same labeled archive appears before or after the current table. Tokenize the table blocks separately so the two archived orders have identical token counts and table contents. The measured current-only contrast includes the effect of added context. Swapping the two labeled blocks isolates order at equal input length.

Evaluate untouched Llama and six seed-matched Llama adapters at the fixed KL weight of 0.5 on HPG RTX PRO 6000. Use native chat formatting, greedy decode, EOS or 384 new tokens, and the same grader as the full MATH-500 comparison. No specific answer marker is requested. The primary effect is the change in KAM minus dot accuracy between archive-last and current-only. Report all three accuracies, pairwise gains and losses, cap and parsing rates, and the archive-first/last interaction. Pair identical items across modes and use title-and-header bootstrap intervals conditional on the three measured seeds. [Evaluator](../../scripts/kam_tabmwp_archive_noise.py) · [GPU driver](../../slurm/kam_tabmwp_archive_noise.sbatch)

## Results on HPG

| Table order | Untouched Llama | KAM | Dot | KAM minus dot (95% interval) |
| --- | ---: | ---: | ---: | ---: |
| Current only | 62.50% | 64.06% | 65.10% | −1.04 [−5.64, +3.70] pp |
| Archive before current | 48.44% | 45.83% | 43.49% | +2.34 [−3.17, +7.92] pp |
| Archive after current | 38.28% | 38.02% | 36.46% | +1.56 [−4.02, +7.41] pp |

The archived table nearest the question reduced KAM by 26.04 points [−34.34, −18.07] and dot by 28.65 [−37.21, −20.43]. The preregistered change in the KAM–dot gap was +2.60 points [−4.80, +10.57]. Moving the archive from first to last at identical lengths lowered accuracy by 7.81 points for KAM and 7.03 for dot. All six seeds-by-arm lost accuracy in that order swap. The intervals for the mean losses from order alone reach slightly above zero. [Paired summary and transitions](../../results/general_benchmarks_tabmwp/llama3-summary.json)

The exact-match source audit found no normalized matches for the 384 checked table and question-plus-table records in the three frozen adapter-training streams. The scorer parsed 2,667 of 2,688 completions and reported zero timeouts or errors. Sixty-one completions reached the token limit. Archive roles were assigned for this controlled stress test on independently sourced tables. [Source audit](../../results/general_benchmarks_tabmwp/source-overlap.json) · [Numeric outcomes](../../results/general_benchmarks_tabmwp/llama3-outcomes.json.gz) · [Scheduler receipt](../../results/general_benchmarks_tabmwp/llama3-scheduler.tsv) · [Summarizer](../../scripts/summarize_kam_tabmwp_archive_noise.py)

### Where the losses came from

A masked review inspected eight hash-selected losses that parsed and finished in both conditions, one per grade and fitted arm. Five responses used values from the archive in an incorrect calculation. One response misread the current table. Two responses gave the correct written conclusion and the parser missed it. This diagnostic sample illustrates ways the errors occur. The sample was selected from losses.

Applying the same grader to the final three nonempty lines of all 2,688 completions changed seven flags. The archive-last loss remained 25.78 points for KAM and 28.65 for dot. The change in their gap was +2.86 points [−4.52, +10.85]. The registered score remains the full-text score. [Masked case labels](../../results/general_benchmarks_tabmwp/masked-archive-review.json) · [Terminal-line paired summary](../../results/general_benchmarks_tabmwp/llama3-terminal-sensitivity.json) · [Terminal-line numeric outcomes](../../results/general_benchmarks_tabmwp/llama3-terminal-outcomes.json.gz)

### Exact-length different-topic control

Choose a different development table with a disjoint set of title words, a different normalized header, and the same grade, row count, column count, and frozen Llama token count as the same-topic archive. Rank donors by a fixed digest, exclude used table contents, and keep the current record and question fixed. This yielded 94 matched questions. Replay the two original batches with the original rows in every unmatched position, and check their token lengths, stop reasons and completion text against the prior runs. Measure changed-source accuracy on the matched rows only. The primary comparison is different-topic minus same-topic accuracy for each arm. The KAM–dot interaction is secondary. [Frozen control manifest](../../results/general_benchmarks_tabmwp/unrelated-manifest.json) · [Tokenizer and selection audit](../../scripts/prepare_kam_tabmwp_length_control.py)
