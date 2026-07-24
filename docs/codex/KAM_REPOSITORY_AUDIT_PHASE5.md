# KAM Repository Audit — Phase V Preconditions

Repository inspected: `Jibby2k1/KAM`
Reference commit: `311a76505de86fb5a254084389d6331da846fe7c`

This file records only the repository findings that must change the next experiment design.

## High-priority validity issues

1. **Current parameter matching is inactive padding.**
   `KAMSequenceModel._add_capacity_padding()` adds an unused zero-valued parameter vector until the requested count is reached. This equalizes reported parameter count but not active capacity, FLOPs, or representational power. No primary comparison may use this as evidence of capacity matching.

2. **The current `S` and `M` labels confound many variables.**
   Width, supports, context length, data length, training steps, batch size, and evaluation budget all change together. In addition, much of the nominal target parameter count can be inactive padding. Scale effects therefore cannot yet be interpreted mechanistically.

3. **The Phase IV factors are not orthogonal.**
   - Prototype `recurring` uses `A→B→A`; `separated` uses `A→B→A→B`, which is also recurring.
   - NARMA and Mackey–Glass use the same `A→B→A` schedule in both conditions and change only regime separation.
   Recurrence, separation, dwell time, and number of transitions must become independent factors.

4. **Training windows are shuffled.**
   Regression loaders use `shuffle=True`. The model therefore does not experience `A→B→A` as an ordered training stream. Current static training can test reusable features across a mixture of windows, but not sequential recurrence, reacquisition, or online memory.

5. **One continuous stream is split by time.**
   With equal schedule segments, train, validation, and test can contain different regime compositions. A nominal improvement may reflect which regime occupies a split rather than generalization. Generate independent train, validation, and test streams with explicitly matched or intentionally shifted schedules.

6. **`RF-b` is not a fully fixed feature map.**
   It freezes only memory keys and values. The memory query projection, score pathway, output projection, backbone, and readout continue learning. It is a fixed random dictionary inside a learned coordinate system, not a fully fixed random-feature baseline.

7. **`DD-b-staged` freezes only keys and values.**
   The query coordinate system continues moving after the bank freezes. Full-memory-path freezing is required to test coordinate mismatch.

8. **Raw route dimensionality changes with support count.**
   The final readout receives `num_heads × num_supports` route coordinates. Increasing supports therefore changes both the bank and the readout feature dimension. Primary comparisons require a fixed projected route dimension.

9. **Batch-averaged NMSE is not global NMSE.**
   `_evaluate()` averages per-batch MSE/variance ratios. Compute global error sums and target variance over the full evaluation set.

10. **Best-checkpoint test semantics must be explicit.**
    Ensure held-out test metrics are computed after reloading the validation-selected checkpoint, not merely from the final training state.

## Scientific implication

The next decisive question is:

> Does a learned support bank outperform an equally rich, genuinely fixed feature basis after active-capacity, route-dimension, data-composition, and training-order controls?

The campaign should distinguish:

- learned semantic/prototype memory;
- fixed random dictionaries in learned coordinates;
- genuinely fixed random features;
- data-initialized fixed kernel centers;
- generic nonlinear feature expansion;
- online readout adaptation.
