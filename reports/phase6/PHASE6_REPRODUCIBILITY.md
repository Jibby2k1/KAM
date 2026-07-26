# Phase 6 reproducibility record

## Scope and reference state

- Authoritative brief: `docs/codex/KAM_PHASE6_SPARSE_SEPARABLE_MEMORY_BRIEF.md`
- Supplied package: `KAM_Phase6_Codex_Package/`
- Scientific reference commit: `2541e09d5dfd37ad162756cf54429f1913b27e0f`
- HPG checkout: `/blue/uf-dsi/rvalle1/KAM_repair_2541e09`
- HPG environment: `/blue/uf-dsi/rvalle1/venvs/kam`
- HPG account/QOS/partition: `uf-dsi` / `uf-dsi` / `hpg-turin`

The workspace contains pre-existing Phase 5 edits and results. Phase 6 changes and artifacts are kept in their own `kam/phase6/`, `configs/phase6/`, `results/phase6/`, and `reports/phase6/` trees.

## Validated gates and runs

| Run | HPG identifiers | Status | Local evidence |
|---|---|---|---|
| Stage 0 validity | `38034590` / `38034986` | 128/128 pass | `results/phase6/stage0/`, `reports/phase6/PHASE6_STAGE0_VALIDITY_REPORT_HPG.md` |
| Stage 1 task-aware screen | `38040026` / `38040027` | 64/64 identity/finite-metric pass; short alternating geometry phases deferred | `results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/` |
| Stage 1 partial superseded run | `38040418` / `38040419` | canceled at 1,315 rows after schedule audit | HPG-only audit root `hpg_runs_full_taskfix2/` |
| Stage 1 corrected full campaign | `38042710` / `38042711` | 3,000/3,000 pass; identity and schedule audits pass | `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/`; `reports/phase6/PHASE6_STAGE1_FULL_REPORT.md` |
| Stage 2 transformer profile | `38049074` / `38049075` | 48/48 pass; exact-manifest and finite-metric audits pass | `results/phase6/stage2_transformer_comparison/hpg_runs_profile_budget1/`; `reports/phase6/PHASE6_STAGE2_PROFILE_REPORT.md` |
| Stage 3 router profile | `38049475` / `38049476` | 32/32 pass; exact-manifest and finite-metric audits pass | `results/phase6/stage3_router_scaling/hpg_runs_profile_scaling1/`; `reports/phase6/PHASE6_STAGE3_PROFILE_REPORT.md` |
| Stage 4 initial profile | `38049583` / `38049584` | superseded after audit found six nonfinite symbolic histories | `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt1/` |
| Stage 4 corrected profile | `38049769` / `38049770` | 48/48 pass after bounded normalized-update repair | `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt2/`; `reports/phase6/PHASE6_STAGE4_PROFILE_REPORT.md` |
| Stage 5 initial profile | `38050204` / `38050205` | superseded after four unsupported Mackey-Glass rows | `results/phase6/stage5_long_training/hpg_runs_profile_long1/` |
| Stage 5 corrected profile | `38050338` / `38050339` | 12/12 pass after bounded dynamics-fixture repair; 4,096-token cap | `results/phase6/stage5_long_training/hpg_runs_profile_long2/`; `reports/phase6/PHASE6_STAGE5_PROFILE_REPORT.md` |
| Stage 6 preparation profile | `38050441` / `38050442` | 12/12 pass at 10M; preparation only, not locked inference | `results/phase6/stage6_confirmation/hpg_runs_profile_confirm1/`; `reports/phase6/PHASE6_STAGE6_PROFILE_REPORT.md` |

The task-aware profile identity audit is saved at `reports/phase6/stage1_mechanism_taskfix3/identity_audit.json`. The reusable audit implementation is `scripts/audit_phase6_run.py`.

## Reproduction commands

```bash
# Build manifests in an environment containing the project dependencies.
python scripts/build_phase6_manifests.py --stage stage1_mechanism --mode profile
python scripts/build_phase6_manifests.py --stage stage2_transformer_comparison --mode profile

# Audit a retrieved HPG run.
python scripts/audit_phase6_run.py \
  --manifest results/phase6/stage1_mechanism/manifests/profile_geometry_fix.jsonl \
  --run-root results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3 \
  --expected 64

# Plan or submit a dependency-gated HPG stage.
scripts/submit_phase6_hpg.sh --plan-only
scripts/submit_phase6_hpg.sh --submit
```

The HPG environment has `pyarrow` installed, so completed aggregates emit true Parquet files rather than mislabeled JSON. HPG task logs are written under `/blue/uf-dsi/rvalle1/logs/phase6/`.

The transformer comparison path records declared total-parameter budgets for 2M/10M/30M rows, measured total and routed active parameters, effective memory capacity, and parameter-match error. The default small-language fixture is the bundled `data/tinyshakespeare.txt` corpus when present; the deterministic one-sentence fixture remains a fallback. `T-KAM-ALT` and `T-KAM-VP` execute distinct transformer optimizer paths and report geometry/algebra update counts. Use `scripts/build_phase6_report.py` after aggregation for reproducible descriptive group tables.

## Interpretation guardrail

Stage 0 is an implementation gate, Stage 1 is a mechanism screen, and Stages 2–6 are bounded profiles. Their row-count, factor-identity, finite-metric, schedule, and matched-resource audits pass for the canonical runs listed above. They still do not provide the paired new seeds, held-out streams/corpora, inferential tests, equivalence margins, or true long-training budget required for a promotion decision.
