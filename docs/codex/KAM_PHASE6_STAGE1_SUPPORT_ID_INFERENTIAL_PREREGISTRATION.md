# Phase 6.2 support-identity Stage 1 inferential preregistration

**Frozen before any row of this campaign is executed.** This is a new prospective inferential campaign. It does not pool with, replace, or rehabilitate the failed original Stage 1 campaign.

## Rationale and sole implementation change

The original Stage 1 validity gate failed because CUDA top-k tie/order behavior was not invariant to matched support permutations. The defect localized to late memory layers, reproduced across three L4 hosts, disappeared on CPU, and was corrected by an opt-in support identity that travels with each matched key/expert permutation. The correction passed a locked 26-checkpoint replay and a prospective 24-row training pilot.

The sole planned implementation difference from the original 168-row design is `router_tie_breaking=support_id`. Architecture, corpus, optimizer arms, endpoints, seeds, checkpoints, precision, and analysis remain unchanged.

## Immutable design

- 168 rows at 50M tokens each on NVIDIA L4 GPUs.
- Eight original arms.
- Four primary arms with 30 paired seeds per arm.
- Four secondary arms with 12 paired seeds per arm.
- Same registered anchor bank, checkpoint schedule, test-loss endpoint, optimizer provenance, freeze checks, and restart checks.
- New content-addressed row identifiers with `supersedes_row_id` linking each row to its design-equivalent original row.

## Validity gate

Scientific interpretation requires all 168 rows to complete and every existing Stage 1 validity check to pass, including strict FP32 matched-permutation identity at `2e-5` and BF16 operational identity at top-1 flip `0.02` and predictive KL `0.001`. Any failure yields `VALIDITY_REPAIR_REQUIRED`; no arm is promoted or pruned.

## Statistical analysis

The estimand, paired seed structure, four primary comparisons, four secondary comparisons, within-family Holm correction, paired randomization tests, paired bootstrap intervals, and ±1% equivalence margins are exactly those in `KAM_PHASE6_STAGE1_CONDITIONAL_DECISION_MEMO.md`.

The original failed campaign may be shown only as historical sensitivity context. It is not included in this campaign's estimates, p-values, intervals, or multiplicity families.

## Downstream decision

The locked conditional actions in `KAM_PHASE6_STAGE1_CONDITIONAL_DECISION_MEMO.md` apply only after this new campaign passes validity. A valid efficacy result still requires the checkpoint-reuse causal and systems gate before Stage 2.
