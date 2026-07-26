# LLM navigation guide

Start here when reviewing this repository:

1. `README.md` — project purpose, model variants, and baseline commands.
2. `EXPERIMENTS.md` — experiment questions and interpretation guardrails.
3. `KAM_Phase4_Data_Regime_Package/KAM_PHASE4_CODEX_DATA_REGIME_BRIEF.md` — authoritative Phase IV scientific specification.
4. `configs/phase4/factorial_screen.yaml` — the currently queued bounded screen.
5. `kam/phase4/manifest.py` — immutable manifest and factor definitions.
6. `kam/phase4/run_array.py` — one-row resumable execution.
7. `kam/phase4/aggregate.py` — metrics, figures, and report generation.
8. `reports/phase4/PHASE4_LLM_HANDOFF.md` — concise context for ChatGPT feedback.

The canonical experiment unit is one manifest row. Outputs live under `results/phase4/` and are keyed by immutable `run_id`; reports and figures live under `reports/phase4/`. Test data is evaluated only after validation-based checkpoint selection. The default Phase IV screen is a development instrument, not confirmatory evidence.

For HiPerGator, inspect `scripts/submit_phase4_hpg.sh --plan-only` before using `--submit`. For local validation, run `pytest -q` and build a small manifest with `python -m kam.phase4.manifest --config configs/phase4/factorial_screen.yaml --output /tmp/kam_phase4_manifest.jsonl`.

Phase V validity gate:
- Authoritative brief: docs/codex/KAM_PHASE5_CODEX_EXECUTION_BRIEF.md
- Repository audit: docs/codex/KAM_REPOSITORY_AUDIT_PHASE5.md
- Supplied campaign specification: configs/phase5/phase5_learned_vs_fixed_features.yaml
- Gate config and runner: configs/phase5/validity.yaml, kam/phase5/
- Gate report and ChatGPT handoff: reports/phase5/

Phase V Stage 2 controlled-regime campaign:
- Authoritative brief: docs/codex/KAM_PHASE5_STAGE2_CODEX_BRIEF.md
- Current repair/rerun handoff: docs/codex/KAM_PHASE5_STAGE2_REPAIR_HANDOFF.md
- Manifests: results/phase5/stage2/manifests/
- Row executor and held-out evaluation: kam/phase5/stage2_run.py
- Gate, statistics, and aggregation: kam/phase5/stage2_gate.py, kam/phase5/stage2_stats.py, kam/phase5/stage2_aggregate.py
- HPG workflow: scripts/submit_phase5_stage2_hpg.sh --plan-only, then --profile, then --submit
- Do not infer Stage 3 scaling, online adaptation, or natural-language conclusions from Stage 2 until the reports and validity checks pass.

Phase VI sparse separable memory:
- Current quality-scale campaign: `docs/codex/KAM_PHASE6_OVERNIGHT_IMPLEMENTATION_GUIDE.md` (start here for the queued four-L4 graph, memory adaptation/freeze semantics, final metrics/figures, and morning commands).
- Authoritative overnight contract/config: `docs/codex/KAM_PHASE6_OVERNIGHT_4XL4_CAMPAIGN.md`, `configs/phase6/overnight_4xl4_campaign.yaml`.
- Timeout repair: the initial Wave 1 gate retained 20/32 valid rows and blocked downstream work after 12 infrastructure timeouts. Read `docs/codex/KAM_PHASE6_OVERNIGHT_TIMEOUT_REPAIR.md`. The replacement graph runs from `38087856` through final report `38087863`; immutable records are `results/phase6/overnight/{job_graph,timeout_repair_job_graph}.json`. Status is queued, not a result.
- Overnight execution/analysis: `kam/phase6/overnight_{manifest,runner,analysis}.py`; controllers `scripts/phase6_overnight_controller.py`, `scripts/build_phase6_overnight_report.py`, and `scripts/submit_phase6_overnight_4xl4.sh`.
- Overnight evidence boundary: do not advise promotion until `results/phase6/overnight/final_summary.json` and all seven `reports/phase6/overnight/OVERNIGHT_*.md` reports exist.
- Authoritative brief: docs/codex/KAM_PHASE6_SPARSE_SEPARABLE_MEMORY_BRIEF.md
- Campaign summary: configs/phase6/phase6_campaign_overview.yaml
- Stage 0 immutable config/manifest builder: configs/phase6/stage0_validity.yaml, kam/phase6/manifest.py
- Stage 1–6 designs: configs/phase6/stage{1..6}_*.yaml and `scripts/build_phase6_manifests.py`
- Decoder/baselines, memory, and optimization surfaces: kam/transformer/, kam/memory/, kam/optimization/
- Data lanes: kam/data/phase6/{dynamics,retrieval,symbolic,language}.py
- Local row execution, gates, statistics, diagnostics, plots, and artifacts: kam/phase6/{run_array,gates,stats,diagnostics,plots,artifacts}.py
- Reproducible manifest/output identity audit: scripts/audit_phase6_run.py
- Full Stage 1 alternating-schedule audit: scripts/audit_phase6_stage1_schedule.py
- Descriptive stage report builder: scripts/build_phase6_report.py
- Local Stage 0 runner/report: kam/phase6/run_stage0.py, reports/phase6/PHASE6_STAGE0_VALIDITY_REPORT.md
- HPG evidence: reports/phase6/PHASE6_STAGE0_VALIDITY_REPORT_HPG.md and results/phase6/stage0/hpg_runs_measured/
- Superseded Stage 1 HPG profile: results/phase6/stage1_mechanism/hpg_runs_profile_final/ (array `38038386`; aggregate `38038387`; retained for audit only; see reports/phase6/PHASE6_STAGE1_TASK_DISPATCH_GAP.md).
- Passing task-aware Stage 1 profile: local output root `results/phase6/stage1_mechanism/hpg_runs_profile_taskfix3/` (HPG array `38040026`; aggregate `38040027`); report: `reports/phase6/PHASE6_STAGE1_TASKFIX_REPORT.md`.
- Superseded partial full Stage 1 audit: HPG array `38040418`, aggregate `38040419`; output root is `results/phase6/stage1_mechanism/hpg_runs_full_taskfix2/`.
- Completed corrected full Stage 1 campaign: HPG array `38042710`, aggregate `38042711`; output root is `results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/`, detailed report is `reports/phase6/PHASE6_STAGE1_FULL_REPORT.md`, and independent audits are under `reports/phase6/stage1_mechanism_full_taskfix3/`.
- Superseded Stage 1 geometry-gap audit: reports/phase6/PHASE6_STAGE1_GEOMETRY_GAP.md and results/phase6/stage1_mechanism/hpg_runs_profile_geometry_gap/.
- HPG workflow: scripts/submit_phase6_stage0_hpg.sh for Stage 0; scripts/submit_phase6_hpg.sh --plan-only|--submit for dependency-gated later profiles.
- Completed bounded Stage 2 transformer profile: HPG array `38049074`, aggregate `38049075`; exact manifest `results/phase6/stage2_transformer_comparison/manifests/profile_hpg_38049074.jsonl`, outputs `results/phase6/stage2_transformer_comparison/hpg_runs_profile_budget1/`, audit/report directory `reports/phase6/stage2_transformer_comparison_profile_budget1/`, and canonical report `reports/phase6/PHASE6_STAGE2_PROFILE_REPORT.md`.
- Completed bounded Stage 3 router profile: HPG array `38049475`, aggregate `38049476`; exact manifest `results/phase6/stage3_router_scaling/manifests/profile_hpg_38049475.jsonl`, outputs `results/phase6/stage3_router_scaling/hpg_runs_profile_scaling1/`, audit/report directory `reports/phase6/stage3_router_scaling_profile_scaling1/`, and canonical report `reports/phase6/PHASE6_STAGE3_PROFILE_REPORT.md`.
- Completed corrected bounded Stage 4 online-adaptation profile: initial HPG array `38049583`/aggregate `38049584` exposed six nonfinite symbolic histories; corrected array `38049769`/aggregate `38049770` passed 48/48. Exact manifest `results/phase6/stage4_online_adaptation/manifests/profile_hpg_38049769.jsonl`, outputs `results/phase6/stage4_online_adaptation/hpg_runs_profile_adapt2/`, audit/report directory `reports/phase6/stage4_online_adaptation_profile_adapt2/`, and canonical report `reports/phase6/PHASE6_STAGE4_PROFILE_REPORT.md`.
- Completed corrected bounded Stage 5 profile: initial HPG array `38050204`/aggregate `38050205` exposed four unsupported Mackey-Glass rows; corrected array `38050338`/aggregate `38050339` passed 12/12. Exact manifest `results/phase6/stage5_long_training/manifests/profile_hpg_38050338.jsonl`, outputs `results/phase6/stage5_long_training/hpg_runs_profile_long2/`, audit/report directory `reports/phase6/stage5_long_training_profile_long2/`, and canonical report `reports/phase6/PHASE6_LONG_TRAINING_REPORT.md`.
- Completed Stage 6 confirmation-preparation profile: HPG array `38050441`, aggregate `38050442`; exact manifest `results/phase6/stage6_confirmation/manifests/profile_hpg_38050441.jsonl`, outputs `results/phase6/stage6_confirmation/hpg_runs_profile_confirm1/`, audit/report directory `reports/phase6/stage6_confirmation_profile_confirm1/`, and canonical report `reports/phase6/PHASE6_CONFIRMATORY_REPORT.md`. This is not final locked evidence.
- ChatGPT handoff and human write-up: reports/phase6/PHASE6_LLM_HANDOFF.md, reports/phase6/PHASE6_REPOSITORY_WRITEUP.md
- Stage reports: reports/phase6/PHASE6_TRANSFORMER_COMPARISON.md, PHASE6_ROUTER_SCALING_REPORT.md, PHASE6_ADAPTATION_REPORT.md, PHASE6_LONG_TRAINING_REPORT.md, PHASE6_CONFIRMATORY_REPORT.md
- Reproducibility record: reports/phase6/PHASE6_REPRODUCIBILITY.md
- Local/HPG Stage 0 evidence is an implementation gate, not a quality result. Later arrays require the HPG gate and paired-seed reporting.
