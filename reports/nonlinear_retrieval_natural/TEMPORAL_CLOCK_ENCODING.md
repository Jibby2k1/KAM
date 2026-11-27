# Clock encoding and candidate compaction for corrected revisions

Signed-log clock features improved radial lookup on synthetic corrections and stable removal of ineligible records improved dot lookup. Neither change recovered the performance of archive-shaped training on held-out release calendars.

The short fit uses fixed 3,200-update procedural streams at candidate depths 16 and 48. Three seeds cross decoder geometry and clock treatment. Linear uses the original scaled offsets. Signed log compresses large absolute offsets while preserving sign. Zeroed mode clears ineligible candidate inputs, leaving their positions visible. In compact mode stable packing moves the eligible records before the readout and rejected records after it, where causal attention cannot use them. The compact readout matched explicit per-case removal to within 3.58e−7 in a CPU check. The experiment supplies eligibility from typed clock fields and measures the remaining learned addressing problem.

| Evaluation | Geometry | Linear | Signed log | Signed log and zeroing | Signed log and compaction |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic corrected, depth 48 | Dot | 3.71% | 11.46% | 72.33% | 71.35% |
| Synthetic corrected, depth 48 | Radial | 14.26% | 65.43% | 28.39% | 49.67% |
| Synthetic corrected, depth 512 | Dot | 0.78% | 1.17% | 20.31% | 18.95% |
| Synthetic corrected, depth 512 | Radial | 1.17% | 28.45% | 2.41% | 4.30% |

Cell values are mean accuracies across three seeds. Every signed-log and signed-log-plus-zeroing outcome was reproduced in the second run. At depth 48 in earlier and later archive slices, compact dot recovered 11.77% and 5.21%, and compact radial recovered 6.98% and 2.60%. Synthetic improvement therefore supplies a diagnostic for eligibility and clock scale. Archive-shaped fit and held-out release groups remain the stronger natural test.

HPG arrays 52987023 and 52988198. [Numeric summaries](../../results/nonlinear_retrieval_natural/temporal_clock_compaction_v1/) · [Training and evaluator](../../scripts/run_kam_temporal_clock_encoding.py) · [Archive-shaped fit](RADIAL_ARCHIVE_TRAINING.md)
