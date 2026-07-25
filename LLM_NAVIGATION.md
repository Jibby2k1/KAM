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
- Manifests: results/phase5/stage2/manifests/
- Row executor and held-out evaluation: kam/phase5/stage2_run.py
- Gate, statistics, and aggregation: kam/phase5/stage2_gate.py, kam/phase5/stage2_stats.py, kam/phase5/stage2_aggregate.py
- HPG workflow: scripts/submit_phase5_stage2_hpg.sh --plan-only, then --profile, then --submit
- Do not infer Stage 3 scaling, online adaptation, or natural-language conclusions from Stage 2 until the reports and validity checks pass.
