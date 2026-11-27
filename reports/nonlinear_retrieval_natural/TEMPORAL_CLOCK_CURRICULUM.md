# Mixed release calendars and signed-log clock features

The same signed-log radial decoder learned two different clock regimes when training batches mixed them. On 48-record panels, it reached 79.82% joint recovery on synthetic corrections and 81.25% on earlier Philadelphia Fed archive groups. Correction-only training reached 65.43% and 0.05%, respectively. Archive-shaped training gave 0% and 83.80%.

Each seed uses 3,200 updates and the same 16- and 48-record batch schedule, sample indices, initialization and learning rate across data regimes. The correction and archive-shaped banks are independent procedural generators. In the mixed arm, half of each batch comes from each generator. The archive-shaped generator supplies clustered releases of the same observation and unique candidate symbols, without using any held-out workbook observations. The test labels are checked by an exact two-clock lookup. The archived slice evaluates answerable requests with typed clock and identity coordinates.

| Geometry and clock | Training examples | Corrected, 48 | Corrected, 512 | Earlier archive, 48 | Later archive, 48 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dot, signed log | Corrections | 11.46% | 1.17% | 0.05% | 0.00% |
| Dot, signed log | Clustered releases | 0.13% | 0.33% | 87.81% | 82.55% |
| Dot, signed log | Mixed | 64.84% | 14.97% | 59.53% | 46.88% |
| Radial, signed log | Corrections | 65.43% | 28.45% | 0.05% | 0.26% |
| Radial, signed log | Clustered releases | 0.00% | 0.65% | 83.80% | 76.56% |
| Radial, signed log | Mixed | 79.82% | 30.53% | 81.25% | 75.52% |
| Radial, linear | Mixed | 11.72% | 1.17% | 50.57% | 37.50% |

Values are means across three matched seeds. Mixed signed-log radial scored 78.71–81.45% on the depth-48 corrections, above the 62.89–70.31% correction-only range. At 512 records, performance remained below one third. The archive shape changes several properties together, including the distribution of release times and competing records. Matching update counts isolates the training mixture as a group intervention. It leaves the individual generator factors open. Zeroing or compacting ineligible inputs helped some dot arms and hurt mixed radial relative to signed log alone.

HPG arrays 52989847 and 52990244 and the original correction controls 52987023 and 52988198. [Matched runs and all clock conditions](../../results/nonlinear_retrieval_natural/temporal_mixed_encoding_v1/) · [Clock analyzer](../../scripts/analyze_kam_archive_clock_encoding.py) · [Initial eligibility diagnostics](TEMPORAL_CLOCK_ENCODING.md)
