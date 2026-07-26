# Phase 6 transformer comparison report

**Status: bounded profile complete and audited.** HPG array `38049074` and aggregate `38049075` completed all 48 Stage 2 rows. The exact submitted manifest, outputs, audit, aggregate metadata, and generated descriptive report are retained under:

- `results/phase6/stage2_transformer_comparison/manifests/profile_hpg_38049074.jsonl`
- `results/phase6/stage2_transformer_comparison/hpg_runs_profile_budget1/`
- `reports/phase6/stage2_transformer_comparison_profile_budget1/`
- `reports/phase6/PHASE6_STAGE2_PROFILE_REPORT.md`

The profile covers 2M, 10M, and 30M target total-parameter budgets across the configured transformer controls and four task fixtures. The 100M language rows remain excluded until the 2M–30M gate passes, as required by the brief.

## Execution and resource checks

- 48/48 rows passed; the identity audit found no missing, extra, duplicate, mismatched, failed, nonfinite, or optimizer-dispatch rows.
- Total-parameter matching was close: mean absolute relative error was 0.214%, with a maximum of 0.5952%.
- Every row completed four optimizer steps and 256 training tokens. These are profile budgets, not convergence evidence.
- `T-KAM-ALT` executed one geometry update and three algebra updates in its short schedule; `T-KAM-VP` executed algebra-only updates. The other controls used four algebra/standard updates and no geometry update.
- The sparse controls expose effective capacity separately from declared capacity: the ordinary KAM rows used 32 effective memory slots, PKM rows 25, and the four-expert MoE control routed through 4 effective experts with top-2 routing.

## Descriptive observations

The profile demonstrates that the matched-budget construction, task dispatch, real bundled language fixture, and operational optimizer paths execute on GPU. Loss/perplexity and timing vary substantially across the intentionally heterogeneous Latin-hypercube rows, so the grouped table in `PHASE6_STAGE2_PROFILE_REPORT.md` is useful for locating candidates and failure modes but not for ranking architectures. In particular, the profile does not provide enough paired seeds, training budget, or held-out evaluation to claim a quality, scaling, throughput, or adaptation advantage.

The next defensible step is to review these resource checks and promote only a small set of Stage 2 candidates/controls into replicated, held-out comparisons before spending on the Stage 3 router-scaling and Stage 4 online-adaptation profiles. No promotion decision is made here.
