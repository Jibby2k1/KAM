# Phase IV reproducibility

- Config: `configs/phase4/factorial_screen.yaml`
- Manifest builder: `python -m kam.phase4.manifest --config configs/phase4/factorial_screen.yaml`
- HPG submission: `scripts/submit_phase4_hpg.sh --submit`
- Array runner: `python -m kam.phase4.run_array --manifest ... --array-index N --run-root ... --device auto --resume`
- Aggregation: `python -m kam.phase4.aggregate --run-root results/phase4/factorial_screen --report-root reports/phase4`
- Expected rows: `96` for the default config.
- Paired unit: task × condition × scale × seed; variants share the seed and generated stream.
