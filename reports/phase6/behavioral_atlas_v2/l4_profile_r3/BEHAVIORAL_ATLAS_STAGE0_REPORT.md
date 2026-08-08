# Phase 6.2 Stage 0 Behavioral Atlas Report

**Stage:** `l4_profile_r3`
**Decision:** `L4_PROFILE_PASS`

- Complete rows: 1/1
- Audit passed: `True`
- Figures: 8 PNG/SVG pairs
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
- permutation_symmetry: `True`
- permutation_operational_stability: `True`
- restart_identity_when_registered: `True`
- actual_nvidia_l4: `True`
- bf16_tf32_fused_adamw: `True`
- bounded_profile_budget: `True`

## Descriptive results

| Arm | Seed | Profile | Validation loss (initial → final) | Test loss | Key drift | Post-freeze drift | Route Jaccard | Support entropy | Dead supports | Memory contribution | Stable rank | tokens/s | VRAM MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_joint_freeze80 | 76000 | bounded_l4_profile | 5.685 → 2.504 | 2.626 | 0.004131 | 0 | 0.1645 | 0.7126 | 0.4672 | 0.002399 | 3.917 | 3005 | 600 |

### Permutation precision diagnostics

```json
[
  {
    "operational_permutation_max_abs_logit_difference": 0.03125,
    "operational_permutation_predictive_kl": 9.533364391245414e-06,
    "operational_permutation_top1_flip_rate": 0.001953125,
    "semantic_permutation_max_abs_logit_difference": 0.0
  }
]
```

## Interpretation boundary

Stage 0 validates measurement, repeatability, runtime, storage, and execution semantics. It is excluded from scientific inference and cannot promote an architecture.
