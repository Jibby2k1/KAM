# When coarsened publication dates make a revision ambiguous

A typed lookup can recover the exact as-of snapshot when it knows the publication day. Converting dates to week or month buckets can make the identity of the latest eligible release ambiguous even when the correct address and complete history are supplied. How often this happens varied substantially across the two audited Eurostat series. Numerical answers remained uniquely determined more often than release identity, since multiple snapshots can report the same value.

## Exact set of possible answers

Within one address and publication bucket, let \(m\) be the number of distinct releases, \(c\) the number of days through the cutoff, and \(r\) the number of days after it. The source publishes at most one release for that address on each day. The number of releases in the bucket available by cutoff can be any integer in

\[
\max(0, m-r) \;\leq\; n_{\mathrm{visible}} \;\leq\; \min(m,c).
\]

If zero is feasible, the latest release can come from the latest older bucket, or the answer may be absent if no older bucket exists. If a positive count is feasible and no within-bucket order survives, any release in the cutoff bucket can be the last visible one. If the order survives, only the release at each possible positive count can win. The same ordering restriction makes just the last ordered record of any older bucket eligible to win. A brute-force check reproduced this result by enumerating all possible placements for 384 small scenarios. This audit assumes arbitrary within-bucket assignments consistent with uniqueness and, in the order-aware policy, the supplied ordering. It does not assume a known release schedule.

## Eurostat replay

The same 1,296 as-of questions from the archived panels were evaluated with full same-address source histories as well as the reduced candidate banks. Daily dates recovered every exact snapshot and value with both policies. Each cell below gives **unique snapshot / unique value**, using full histories and retaining only the coarse date bucket.

| Panel | Questions | Week | Fortnight | Calendar month |
| --- | ---: | ---: | ---: | ---: |
| Unemployment, 2021–24 | 576 | 290 / 497 | 205 / 474 | 158 / 443 |
| Industrial production, 2021–24 | 576 | 556 / 569 | 398 / 438 | 257 / 306 |
| Unemployment, first half 2025 | 72 | 57 / 68 | 34 / 63 | 6 / 55 |
| Industrial production, first half 2025 | 72 | 71 / 72 | 65 / 68 | 7 / 23 |

Retaining true within-month order raised exact unemployment resolution from 158 to 328 on 576 original questions and from six to 21 on 72 later questions. It had little effect on industry. The cutoff still falls inside the same bucket, so sequence order alone cannot recover whether a specific release came before it. The order-aware month policy can outperform the week policy because different bin boundaries align differently with the cutoffs. A reader can abstain on all ambiguous records while providing correct answers for the uniquely determined subset under this observation model.

The history and cutoff labels come from the [official vintage fixtures](RELEASE_PHASE_TRAINING.md). The conservative counts are exact for the specified metadata loss and the observed source. The 2025 windows cover only two quarters. [Executable oracle and exhaustive validation](../../scripts/audit_kam_revision_clock_resolution.py) · [All paired resolution outcomes](../../results/nonlinear_retrieval_natural/revision_clock_resolution_v1/)
