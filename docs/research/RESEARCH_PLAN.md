# KAM research plan

This plan advances the matched KAM and dot residual comparison, the natural retrieval studies, and the temporal-memory experiments. Each study has a frozen comparison, a stated decision, and a reproducible result. Updated benchmark scores go in the [scorecard](BENCHMARKS.md). Reports record the measurements that justify a change in direction.

## Current evidence and decision

The pretrained adapter factorial established that KL can recover the untouched model's correct answers while reducing gains on previously missed items. The strength that worked best depended on the backbone. A held-out two-format rule selected KL 4 for SmolLM2 and 0.5 for Llama 3 at a learning rate of 0.00008. On the MATH-500 integer subset, KAM led under one marker on SmolLM2, dot led under that marker on Llama, and the two adapters tied or nearly tied under the alternate marker. The Llama both-marker result also reflected whether the required output appeared. This leaves a direct question about the same frozen checkpoints under a neutral prompt with mathematical equivalence scoring. [Matched factorial](GENERAL_BENCHMARKS_FACTORIAL.md) · [Two-marker evaluation](MATH_KL_TRANSFER.md)

## 1. Full MATH-500 under one neutral prompt

| Design element | Locked choice |
| --- | --- |
| Models | SmolLM2-1.7B-Instruct and UF-hosted Meta Llama 3 8B Instruct, each with an untouched reference |
| Adapters | Parameter-matched KAM and dot at the previously selected KL weight and learning rate |
| Seeds | 731201, 731211, 731221 |
| Fixture | All 500 items from the pinned Hugging Face H4 MATH-500 test release |
| Prompt | The verbatim problem followed by “Solve the problem. Explain your reasoning and state your final answer clearly.” |
| Decoder | Greedy, native chat templates, 1,024 generated tokens, EOS or limit, length-sorted batches of 64 |
| Grader | Math-Verify 0.9.0 with a pinned parser and SymPy, shared settings across all models and arms |
| Primary endpoint | Verified mathematical answer on the full 500-item set |
| References | Untouched backbone, plus the earlier integer subset evaluated under its two original protocols |

Gold answers are parsed separately from completions and must all pass a self-equivalence check. The primary grader uses the first extraction match, allows bare arithmetic and LaTeX, and uses symbolic equivalence with strict variable matching. Keep generation and grading separate. Preserve raw generations in restricted experiment storage for audits, and release numeric outcomes and hashes. The full fixture includes the 311 previously tested integer-answer problems and 189 additional problems.

**Analysis.** Report KAM, dot and untouched accuracy per backbone. Compute paired KAM–dot differences on identical questions, with item-bootstrap intervals conditional on the three measured seeds and the three seed-specific differences. Show parse rates and cap stops. Recompute outcomes at the first 384 generated tokens using each full trajectory, then examine the integer and noninteger subsets and subject/difficulty counts. Treat subgroup and parser sensitivity as descriptive because the campaign has two backbones and many correlated comparisons.

**Validity checks.** Verify the fixture, checkpoint, selection, adapter parameter, hardware and grader identities before aggregation. Confirm common prompt lengths and ordering within backbone. Verify parse coverage of all 500 golds. Check held-out adapter-training files for exact prompt overlap. Include clean and tricky authored grader examples such as equivalent decimals and fractions, reordered sets, wrong interval boundaries and intermediate results. Audit a fixed disagreement sample with the arm labels masked before inspecting answers. Track extraction errors separately from invalid reasoning. The primary score and interpretation retain the locked settings.

**Decision.** A practically important and consistent difference on full, integer and noninteger slices warrants a second independent natural-math fixture at the same checkpoint settings. A difference concentrated in the prompt, token budget, parser or one subject warrants a controlled explanation before extending any architecture claim. Opposite signs by backbone motivate reporting the interaction and testing a third permitted backbone only after resolving the relevant mechanism.

## 2. Separate output extraction, decoding budget and adapter behavior

1. **Output extraction.** On the same frozen generations, compare the shared grader with a terminal-three-lines sensitivity check and inspect changed pairs. Record when an unanchored number is an intermediate value, when a later response contradicts the chosen candidate, and when equivalent semantic answers have different surface forms. Use a fixed hash order for sampling disagreements. A useful follow-up scorer should preserve transparent rule-based grading and document how unscorable outputs enter the denominator.
2. **Decoding budget.** Compare the 384-token prefix with the 1,024-token outcome on the same rollout. Record EOS and cap frequency, gains and reversals per item, and budget-stop count by backbone and arm. Cross-check a small independently generated 384-token panel to verify prefix replay, then use it for cost-aware reporting.
3. **Where the residual acts.** If the score gap persists on co-parsed items and under both budgets, run checkpoint-only per-layer residual interventions. Select scale or intervention settings only on development data. Use matched layer positions and run the untouched control through the same evaluator.
4. **Transfer.** Use an independent natural math source with separately documented provenance, overlap, answer types and difficulty. Freeze it before benchmarking. Prefer an evaluation with enough items for a paired effect and coverage across answer types. Match generation budgets and use the same grader. Keep the initial MATH-500 results fixed as the first measurement. Do not retune KL from test outcomes.
5. **Retention.** For every adapted run, publish the net number of previously correct items lost and previously missed items gained. Compare the KL/accuracy frontier on raw text, answer-cue distributions and marker-free mathematical assessment.

### Locked prompt and grading intervention

Use 512 unused GSM8K source-training questions, selected by SHA-256 rank after excluding the prior development set, duplicate prompts, and exact matches to earlier benchmark prompts. Check exact overlap against the frozen adapter-training corpora before analysis. Apply the previously selected KL weights to both arms and include the untouched references. Each checkpoint receives the neutral prompt and two versions with one added sentence requesting either `####` or `Answer:`. Keep batch membership, greedy decoding, the 384-token cap and EOS stopping common across prompts.

First score all three generation conditions with the same pinned equivalence grader. This tests the effect of the prompt request. Then rescore the two marked conditions with the corresponding strict marker and exact number rule on the same generated text. This tests the effect of the grading rule. Report paired changes in the KAM–dot contrast for each operation and compare both with the neutral accuracy on this independent set of questions. Freeze settings before the GPU jobs and retain both mechanisms in the report even if one effect is small.

### Locked replication of answer-type differences

On the full MATH-500 test, Llama dot led on integer answers and KAM led on noninteger answers in each seed. A descriptive reweighting to common subject-by-level cells gave KAM–dot differences of −4.11 points on integers and +2.71 points on nonintegers. This motivates replication on disjoint questions from the same original MATH source. [Full benchmark](FULL_MATH500.md)

Use the 4,500 original MATH test IDs that the PRM800K release assigns outside its MATH-500 test. The pinned file has one duplicate test ID with the same question and answer, which is counted once. Exclude exact normalized duplicate or previously evaluated prompts and answers that fail the frozen gold check. Hash-rank IDs within each subject, difficulty and answer-type cell. Take up to seven equal-count integer and noninteger pairs per cell, yielding 239 pairs across 35 cells. Reuse the matched frozen checkpoints and neutral prompt.

The primary replication endpoint is Llama's KAM–dot difference on noninteger answers minus that difference on integer answers. Positive values are the direction observed in MATH-500. Bootstrap the 239 matched item pairs conditional on the same three seeds. Report SmolLM2 as a backbone comparison. Preserve the 1,024-token neutral prompt, grader and untouched references. Interpret the study as replication on disjoint items from the same benchmark family. [Pinned source](https://github.com/openai/prm800k/blob/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/train.jsonl)

## 3. Test routing on evidence with genuine alternatives

| Question | Next controlled comparison | Primary measurement |
| --- | --- | --- |
| Can one-pass addressing recover a supporting pair when a linked distractor is present? | On a frozen fresh split, compare title-link reranking, learned pair scoring and learned sequential retrieval with the same corpus candidates. | Both supports recovered and joint exact match by complete/incomplete and linked/unlinked pools |
| Does access to an expanded corpus help beyond reranking? | Freeze the conditioned second-hop union, compare at matched end-to-end latency and account for both search passes. | Added support and net joint exact match on previously incomplete pools |
| Does the reader need the complete chain? | Keep two plausible terminal passages visible and swap which verified early chain supports the endpoint. | Paired endpoint accuracy and sensitivity to early evidence substitution |
| Can updating memory help across revision releases? | Replay actual release histories with causal cutoff admission. Hold alias resolution and answer feedback constant while varying retrieval and memory update. | Current-version match, stale-return rate, expected calibration error and coverage at a fixed precision threshold |

Keep evaluation questions and linked distractors separate from fitting and calibration. Report corpus coverage, selection conditional on coverage, and final answer performance separately. Verify that normalized title aliases and temporal facts existed at the recorded cutoff. Evaluate alias abstention, ambiguous collision resolution, and output validity as separate mechanisms. These comparisons use the previously published routing and revision fixtures as references. [Evidence and reproducibility index](RESULTS.md)

## 4. Make efficiency a matched measurement

- **Model architecture.** Record active parameters, trainable parameters, residual cache bytes, decode length, and measured load, prefill and decode time. Compare at fixed output budget and record throughput conditional on equivalent outputs. Add input-length and batch-size buckets where they can change the result.
- **Software.** Retain the exact teacher-cache, shared-prefix and router switches as opt-ins. Compare faster kernels with the prior path on per-token parity or output stability, whichever the experiment needs. A change in numerical precision or batch shape requires a separate fidelity panel.
- **Hardware.** Match training hardware within seed. Schedule independent checkpoint evaluations in parallel, reuse frozen checkpoints and a shared grader, and reserve GPU time for generation. Report elapsed queue-inclusive time separately from summed GPU work. When the queue is saturated, finish a complete paired block before starting an exploratory branch.
- **Prioritization.** First measure performance or output changes that could alter the scientific conclusion. Spend additional GPU time on a diagnostic only when the preceding result identifies what it can decide. Reuse per-item records for analysis that does not require a new model call.

## Publication checkpoints

| Checkpoint | Evidence required | Publish |
| --- | --- | --- |
| Protocol locked | Pinned fixture, shared grader audit, trained-checkpoint identities and stated decisions | Research plan and reproducible protocol on GitHub |
| Primary results verified | All matched runs and references complete, paired summaries replay, public artifacts contain numeric outcomes | Scorecard and study report on GitHub, short result on the KAM course and the personal project page |
| Mechanism checked | Fixed diagnostic cohort, paired item changes and resource measurement | Report update and a website note if the result changes the conclusion |
| Transfer replicated | An independent fixture, locked procedure and complete matched results | Scorecard, public report and website update with the supported scope of the result |

During the November 27 evening session, prioritize the full MATH-500 result, the budget and extraction checks, and the two website deployments. Follow with the smallest matched GPU or CPU diagnostic that can change the conclusion. At the 10 PM Eastern handoff, list verified findings, publication status, and any jobs still running.
