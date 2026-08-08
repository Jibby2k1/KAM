# Phase 6.2 Stage 0 Behavioral Atlas Report

**Stage:** `l4_profile`
**Decision:** `L4_PROFILE_BLOCKED`

- Complete rows: 1/1
- Audit passed: `False`
- Figures: 4 PNG/SVG pairs
- Forecast storage with headroom: 0.00 GiB
- Forecast L4 GPU-hours: 0.02

## Audit

- all_rows_present: `True`
- all_rows_passed: `True`
- initial_states_identical_within_seed_and_identity_group: `True`
- anchors_identical_within_seed_and_size: `True`
- sample_order_identical_within_seed_and_budget: `True`
- finite_metrics: `True`
- executable_optimizer_provenance: `True`
- standard_trace_complete: `True`
- freeze_integrity: `True`
- permutation_symmetry: `False`
- restart_identity_when_registered: `True`
- actual_nvidia_l4: `True`
- bf16_tf32_fused_adamw: `True`
- bounded_profile_budget: `True`

## Profile metrics

```json
{}
```

## Interpretation boundary

Stage 0 validates measurement, repeatability, runtime, storage, and execution semantics. It is excluded from scientific inference and cannot promote an architecture.
