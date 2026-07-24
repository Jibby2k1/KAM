# Phase V reproducibility

- Config: configs/phase5/validity.yaml.
- Manifest: results/phase5/manifests/validity.jsonl.
- HPG submission: scripts/submit_phase5_hpg.sh --submit.
- Aggregation: python -m kam.phase5.aggregate --run-root results/phase5/validity_gate --report-root reports/phase5 --expected 24.
- The full Phase V campaign must not be queued until validity_checks.json reports passed: true.
