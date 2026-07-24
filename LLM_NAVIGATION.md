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
