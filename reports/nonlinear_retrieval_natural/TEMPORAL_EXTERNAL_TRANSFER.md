# Clock encoding transfer to the Dallas Fed release archive

Mixed training with signed-log radial clocks reached 42.44% joint lookup accuracy on held-out Dallas Fed release sequences at 48 records. The archive-shaped signed-log radial arm reached 43.95%, with a paired interval spanning zero for the difference. The best specialized arm in this test, archive-shaped signed-log dot, reached 60.59%.

The fit uses the same 3,200-update procedural stream, learning rate and architecture as the earlier clock study. Its three training regimes contain corrections, clustered releases, or half of each per batch. No Dallas observation enters fitting. The independent Dallas fixture uses 292 country-quarter groups from four GDP workbooks. Each group supplies two answerable vintage cutoffs and two independently assigned symbol encodings at each depth. The address, dates and records are supplied as typed inputs. Every fit reproduced the original procedural and Philadelphia outcomes. The experiment also triples relative publication offsets at evaluation while preserving the candidate identities, target record, eligibility and release order.

| Geometry and clock | Training source | Observed joint, 48 | Publication offsets ×3, 48 |
| --- | --- | ---: | ---: |
| Dot, signed log | Corrections | 0.03% | 0.09% |
| Radial, signed log | Corrections | 0.00% | 0.00% |
| Dot, signed log | Clustered releases | 60.59% | 60.39% |
| Radial, linear | Clustered releases | 55.14% | 55.79% |
| Radial, signed log | Clustered releases | 43.95% | 43.32% |
| Dot, signed log | Mixed | 23.34% | 28.68% |
| Radial, linear | Mixed | 8.36% | 10.22% |
| Radial, signed log | Mixed | 42.44% | 48.46% |

At observed offsets, signed log improved mixed radial by 34.08 percentage points over linear (95% quarter-and-seed interval 28.43 to 39.73). Relative to archive-shaped signed-log radial, mixed signed-log radial was −1.51 points (−12.81 to 9.81). Tripling the publication offsets improved mixed signed-log radial by 6.02 points (3.03 to 9.97), despite leaving the correct lookup unchanged. The result supports partial transfer of the mixed model and exposes continued sensitivity to the representation units. The same mixed model recovered 79.82% on synthetic corrections and 81.25% on the earlier Philadelphia panel at this depth.

HPG arrays 52992482, 52992483 and 52992484. [Matched numerical records](../../results/nonlinear_retrieval_natural/temporal_external_encoding_v1/) · [Dallas fixture and source provenance](OECD_REVISIONS.md) · [Mixed training](TEMPORAL_CLOCK_CURRICULUM.md)
