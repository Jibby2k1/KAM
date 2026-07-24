# Phase V reproducibility

- Config: configs/phase5/pilot.yaml.
- Manifest: results/phase5/manifests/pilot.jsonl.
- Runner: kam/phase5/pilot_run.py.
- HPG submission: scripts/submit_phase5_pilot_hpg.sh --submit.
- Aggregation: python -m kam.phase5.pilot_aggregate --run-root results/phase5/pilot --report-root reports/phase5 --expected 144.
- The pilot is a Stage 1 screen; Stage 2 factorial search requires review of the generated handoff.
