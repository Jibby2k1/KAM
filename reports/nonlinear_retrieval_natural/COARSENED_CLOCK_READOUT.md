# Chronological input visibility and output feasibility

On the original Eurostat unemployment panel, replacing exact publication days with the start of their week reduced learned dot snapshot recovery from 2,766 to 1,364 of 3,456. It exposed an authoritative target-address future candidate on 1,290 queries. Conservative week-end time admitted none, but the unconstrained classifier still emitted four payload IDs associated with future records. Constraining output to records in the eligible prefix removed every observed future emission with daily or week-end dates.

Three frozen guarded dot and radial fits are replayed on the same native vintage questions and candidate sets. All parties see the same typed addresses, publication-week labels, effective clocks and alias metadata. Only the representative publication value given to the fact clock changes from exact day to the start, middle or end of its week. The ground truth uses the source's original publication day. The report separately audits prefix visibility and the date attached to the record code returned by the decoder. The decoder can predict a code that was never in its prefix, even when causality prevents attention to that hidden record. The daily outcomes and every original coarsened outcome replayed exactly before adding visibility masks.

The output policies share the learned alias. One drops any value absent from the admitted records for that alias and workspace. Another masks the logits to those eligible values before choosing. An executable control returns the unique latest value among eligible records at that address and abstains on conflicting ties. The exact selection under an imputed clock can return an earlier true snapshot. Date granularity determines what the guard can safely admit.

| Original depth-48 dot panel | Input date and output policy | Accepted / 3,456 | Joint exact | Future record codes returned |
| --- | --- | ---: | ---: | ---: |
| Unemployment | Exact day, unrestricted | 3,456 | 2,766 | 3 |
| Unemployment | Exact day, mask logits | 3,421 | 2,811 | 0 |
| Unemployment | Week start, unrestricted | 3,456 | 1,364 | 1,419 |
| Unemployment | Week middle, unrestricted | 3,456 | 2,670 | 107 |
| Unemployment | Week end, unrestricted | 3,456 | 2,727 | 4 |
| Unemployment | Week end, mask logits | 3,435 | 2,756 | 0 |
| Unemployment | Week end, latest eligible | 3,435 | 2,759 | 0 |
| Industry | Exact day, unrestricted | 3,456 | 3,152 | 4 |
| Industry | Week start, unrestricted | 3,456 | 3,012 | 136 |
| Industry | Week end, unrestricted | 3,456 | 3,316 | 5 |
| Industry | Week end, latest eligible | 3,446 | 3,323 | 0 |

With exact days, masking eliminates future codes without excluding an exact joint answer. On weekly midpoints, a mask can still emit a future record because the input imputation declared it available. Conditional output eligibility alone cannot repair an optimistic clock. Unemployment has 1,740 formally unique snapshots under week-only publication buckets in these repeated comparisons, compared with 3,336 for industry. A model can guess above these deterministic-uniqueness counts. For each series, the table aggregates three fits and two record symbolizations of the same questions.

HPG L4 arrays 53019080 and 53019636. [Per-query replay and output-policy counts](../../results/nonlinear_retrieval_natural/coarsened_clock_readout_v2/) · [Exact weekly feasibility](REVISION_CLOCK_RESOLUTION.md) · [Calendar priors and uncertainty](REVISION_CALENDAR_PRIOR.md)
