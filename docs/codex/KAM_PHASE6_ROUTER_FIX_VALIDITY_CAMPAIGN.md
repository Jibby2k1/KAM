# Phase 6.2 support-identity router validity campaign

## Status and scope

This campaign is preregistered after the original Stage 1 validity failure and its localization to CUDA tie/order behavior. It is prospective but noninferential. It does not revise, replace, or rehabilitate the original Stage 1 campaign.

## Gate A: locked checkpoint replay

Replay the support-identity policy on the immutable 26-checkpoint audit manifest. The pilot may start only if all conditions pass:

- all 26 rows execute and preserve anchor identity;
- strict FP32 matched permutation identity passes 26/26 at `2e-5`;
- BF16 operational identity passes 26/26 at top-1 flip `0.02` and predictive KL `0.001`;
- unpermuted stable-versus-legacy prediction drift remains within those same operational tolerances for every row.

## Gate B: 24-row training pilot

Use all eight Stage 1 arms and the three lowest seeds shared by all arms. Train each row for 10M tokens with `router_tie_breaking=support_id`. This is an implementation-validity pilot, not an efficacy test.

The pilot passes only if:

- all 24 rows execute;
- strict FP32 and BF16 operational matched-permutation checks pass 24/24;
- restart state hashes match 24/24;
- median throughput is at least 80% of the corresponding original Stage 1 rows;
- no row changes 10M-token validation loss by more than 5% relative to the corresponding original Stage 1 trace.

A passing pilot authorizes planning—not automatic execution—of a newly versioned full Stage 1 inferential campaign. A failed gate stops the sequence.
