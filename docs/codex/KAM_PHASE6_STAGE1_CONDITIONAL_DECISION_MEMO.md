# Phase 6.2 Stage 1 conditional decision memo

**Frozen before the complete Stage 1 report is inspected.** This memo defines how the registered Stage 1 outputs map to follow-up work. It does not alter the 168-row manifest, endpoint, seeds, multiplicity families, or fixed-sample analysis.

## Evidence boundary

Use only the complete exact-manifest Stage 1 report. Do not interpret interim losses, replace seeds, extend Stage 1, or use Stage 2 data to revise these rules. Infrastructure failures may rerun only their original content-addressed rows.

A scientific interpretation requires every Stage 1 validity check to pass: 168/168 passing rows, exact identities and pairing, finite complete traces, optimizer provenance, restart and freeze integrity, the registered anchor size, permutation semantics, and the BF16 operational gate. If any check fails, the outcome is `VALIDITY_REPAIR_REQUIRED`; no arm is promoted or pruned.

## Fixed statistical classifications

The estimand is the paired log held-out-loss ratio `log(first / second)` at 50M tokens. Negative values favor the first arm. Holm correction remains separate within the primary and secondary families.

For each registered comparison:

- `FIRST_PRACTICALLY_BETTER`: Holm-adjusted p <= 0.05 and the bootstrap 95% CI lies entirely below -1% relative loss.
- `SECOND_PRACTICALLY_BETTER`: Holm-adjusted p <= 0.05 and the bootstrap 95% CI lies entirely above +1% relative loss.
- `EQUIVALENT_WITHIN_1_PERCENT`: the paired bootstrap-TOST 90% CI lies wholly inside the asymmetric log margins `log(0.99)` and `log(1.01)`.
- `DIRECTIONAL_OR_UNCERTAIN`: every other outcome. Lack of significance is not equivalence.

Report effect estimates, both intervals, win rates, standardized paired effects, and raw/Holm p-values regardless of classification.

## Primary comparison actions

| Registered comparison | Result pattern | Locked interpretation and action |
|---|---|---|
| no-freeze / fixed | First practically better | Learned geometry has end-to-end lifecycle value. Retain no-freeze as a Stage 2 anchor, subject to causal branch-use checks. |
| no-freeze / fixed | Second better or equivalent | Prefer fixed geometry on parsimony unless freeze-80 independently beats fixed. Do not promote no-freeze. |
| freeze-80 / fixed | First practically better | Early geometry learning followed by stabilization has value. Retain freeze-80 as a Stage 2 anchor, subject to causal checks. |
| freeze-80 / fixed | Second better or equivalent | Do not promote freeze-80 over fixed geometry. |
| freeze-80 / no-freeze | First practically better | Late key motion is harmful. Prioritize freeze timing, smooth decay, lower geometry LR, anchoring, and trust regions. |
| freeze-80 / no-freeze | Second practically better | Continued geometry learning is useful. Prioritize no-freeze, smooth decay, and controlled continuous updates. |
| freeze-80 / no-freeze | Equivalent | Choose the simpler/cheaper policy after resource comparison; do not claim freezing improves quality. |
| ALT8 / joint freeze-80 | First practically better | Algebra/geometry separation matters. Retain alternating schedules and their geometry-LR interaction in Stage 2. |
| ALT8 / joint freeze-80 | Second better or equivalent | Joint optimization is the default; alternating schedules receive no promotion claim. |

If neither learned arm practically beats fixed keys, the full learned-geometry Stage 2 search is not automatic. Run checkpoint-reuse causal diagnostics first and prefer fixed KAM for later matched-architecture comparisons.

## Secondary comparison actions

Secondary results are interpreted only after a valid primary family and with their own Holm correction.

- Freeze-25 or freeze-50 beating freeze-80 promotes earlier stabilization as a candidate and a possible compute-saving policy.
- ALT32 beating ALT8 promotes more conservative geometry frequency; the reverse retains ALT8.
- Cosine decay beating no-freeze promotes smooth stabilization without claiming that a hard freeze is necessary.
- Equivalent secondary policies are resolved by throughput, VRAM, implementation complexity, and checkpoint reuse—not by uncorrected means.

## Causal and systems gate before Stage 2

A statistically favorable learned arm must also show functional memory use in a checkpoint-reuse audit before expensive search:

1. memory-branch deletion changes held-out predictions or loss materially;
2. restoring keys or experts to initialization has a paired functional effect;
3. matched key/expert permutation remains invariant while mismatched shuffles are disruptive;
4. support deletion has a sensible dose-response;
5. the candidate is not dominated on throughput, VRAM, or storage by the simpler comparator.

These diagnostics reuse Stage 1 checkpoints and do not retrain. If the branch is causally inactive, the outcome is `STOP_OR_REDESIGN_MEMORY_BRANCH`, even if a raw arm mean looks favorable.

## Conditional next stages

- `PROCEED_STAGE2`: at least one learned-versus-fixed primary comparison is practically favorable, the validity gate passes, causal memory use is present, and the candidate is not systems-dominated.
- `FIXED_KAM_ONLY`: learned policies lose or are equivalent to fixed keys, but fixed KAM remains worth comparing with matched Transformer, PKM, MoE, and memory-token controls.
- `TARGETED_MECHANISM_REVISION`: routing collapse, useless key drift, weak experts, or co-adaptation failure is causally localized. Write a new bounded preregistration for that mechanism before fresh evaluation.
- `REPLICATE_UNCERTAIN_EFFECT`: an important effect is directional but imprecise. Do not add Stage 1 seeds; use a new preregistered replication with fresh paired seeds.
- `STOP_OR_REDESIGN_MEMORY_BRANCH`: learned and fixed memory lack practical/causal value at the tested scale.

If Stage 2 proceeds, generate its manifest only after this decision is recorded. Screening remains 72 constrained cells x 3 fresh seeds at 10M tokens, validation-only Pareto selection, at most six frozen configurations, and 12 fresh paired confirmation seeds at 50M tokens. Any pruning or factor amendment must be documented before Stage 2 outputs exist.
