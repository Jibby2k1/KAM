# Phase 6.1 Parameter-Dynamics Report

**Stage:** `pilot`
**Decision:** `PILOT_PASS`

- Complete rows: 10/10
- Audit passed: `True`
- Required figures: 8/8 (PNG and SVG)

## Audit

- all_rows_present: `True`
- all_rows_passed: `True`
- initial_states_identical_within_seed: `True`
- finite_metrics: `True`
- all_parameter_groups_present: `True`
- fixed_keys_unchanged: `True`
- freeze_event_order_valid: `True`
- freeze_integrity: `True`

## Locked estimands

- fixed_keys: stabilization ratio `None`; 95% CI `[None, None]`; nearly-frozen pass `False`.
- learned_joint_freeze50: stabilization ratio `0.9101988087880108`; 95% CI `[0.8997940163441646, 0.9206036012318571]`; nearly-frozen pass `False`.
- learned_joint_freeze80: stabilization ratio `0.9101988087880108`; 95% CI `[0.8997940163441646, 0.9206036012318571]`; nearly-frozen pass `False`.
- learned_alt8_freeze80: stabilization ratio `1.1604717331866872`; 95% CI `[1.1545224915882106, 1.1664209747851637]`; nearly-frozen pass `False`.
- learned_joint_no_freeze: stabilization ratio `0.5321635432904898`; 95% CI `[0.527299186445771, 0.5370279001352085]`; nearly-frozen pass `False`.
- learned_joint_freeze80 vs fixed keys at 40M: relative validation change `None`; Holm p `None`.
- learned_alt8_freeze80 vs fixed keys at 40M: relative validation change `None`; Holm p `None`.
- Freeze-80 vs no-freeze at final test: relative change `0.0058148767580537`; paired p `0.5`.

## Interpretation boundary

Pilot rows validate instrumentation only. Main rows test parameter stabilization and freeze effects; neither stage can overturn confirmation v2.
