# Phase V validity audit

Gate: PASSED. Completed 24 of 24 rows with 0 failure artifacts.

## What this gate establishes

This is a precondition check for the Phase V learned-versus-fixed-feature campaign. It verifies implementation semantics; it does not establish that learned supports beat fixed features.

## Checks

- expected_rows_complete: PASS
- no_failure_artifacts: PASS
- zero_padding_primary_rows: PASS
- fixed_route_dimension: PASS
- best_checkpoint_test_reload: PASS
- global_nmse_present: PASS
- independent_split_streams: PASS
- row_validity_checks: PASS

## Evidence

- results/phase5/validity_gate/all_metrics.csv contains one row per completed validity run.
- results/phase5/validity_gate/validity_checks.json contains the machine-readable gate.
- figures/validity_nmse.png shows held-out global NMSE by task, variant, and protocol.

## Interpretation

The full Phase V pilot remains blocked unless every gate check passes. Even a passing gate only authorizes the next implementation stage; it is not evidence for a final decision.
