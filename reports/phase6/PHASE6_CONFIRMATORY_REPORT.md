# Phase 6 locked confirmation report

**Status: confirmation-preparation profile complete; no locked promotion claim established.** HPG array `38050441` and aggregate `38050442` completed all 12 configured profile rows at the matched 10M budget. The exact manifest, outputs, identity audit, aggregate metadata, and descriptive report are retained under:

- `results/phase6/stage6_confirmation/manifests/profile_hpg_38050441.jsonl`
- `results/phase6/stage6_confirmation/hpg_runs_profile_confirm1/`
- `reports/phase6/stage6_confirmation_profile_confirm1/`
- `reports/phase6/PHASE6_STAGE6_PROFILE_REPORT.md`

## Preparation evidence

- 12/12 rows passed identity, completeness, finite-metric, and dispatch audits.
- The three configured claims each have four profile rows: KAM versus widened FFN, KAM versus conventional MoE/PKM controls, and alternating versus joint optimization.
- All rows used four steps and 256 training tokens. Mean absolute relative parameter-budget error was 0.235%, with a maximum of 0.522%.
- The one `T-KAM-ALT` profile row executed one geometry and three algebra updates; the other rows are controls or algebra-only paths.

This is not the locked confirmatory evidence required by the brief. The profile has too few paired new seeds, no held-out stream/corpus analysis, and no inferential tests or equivalence margins. It prepares the execution surface and identifies candidate contrasts; no outcome is selected and no configuration is promoted.
