# Monthly event counts with matched false dates

On 288 positive TimelineQA count questions, changing eight wrong-month events to display a date in the requested month reduced dense top-eight complete evidence recovery from 278 to 97. An authoritative month filter kept 286 complete evidence sets under both displays. Counting literal matches using the authoritative month recovered every count, including 92 zero-count controls.

The fixture uses two author-provided question templates over exercise and lunch events in 24 seeded lifelogs at three densities. Each answer is an exact count of matching emitted event IDs. A SQLite month-and-field query and an independently filtered set of structured records agree on every answer. Positive cases contain two to twelve supporting records. Each candidate pool retains all events of the relevant type in the requested month, eight same-field events from other months, and other same-type background until the pool reaches 128 events or the source is exhausted.

The sham retains each event's authentic displayed date. In the collision, the eight wrong-month events display a date in the requested month. Dates in their rendered bodies also change where present. The underlying event IDs, source dates, action and other fields stay fixed. Every changed record remains an invalid answer under the authoritative source clock. This is a controlled synthetic metadata intervention.

| Evidence policy | Top eight, authentic display | Top eight, collision | Top 16, authentic display | Top 16, collision |
| --- | ---: | ---: | ---: | ---: |
| Frozen E5 dense | 278 | 97 | 287 | 283 |
| E5 restricted by displayed month | 286 | 107 | 288 | 287 |
| E5 restricted by authoritative month | 286 | 286 | 288 | 288 |

These counts report full evidence on the 288 positive questions. For the displayed-month top-eight policy, the paired loss was 62.15 points, with a persona-bootstrap interval of 54.17 to 70.14 points. With more rank slots, most true events fit alongside the eight decoys. A fixed count bound is a separate issue. Counting literal same-field matches over the displayed month overcounted by exactly eight on all 380 collided cases. Counting over the authoritative month preserved the exact same eligible event set and count on all 380. The structured filter assumes access to trusted source dates and the known requested field. It provides an exact control for generated records.

HPG L4 array 52986844 evaluated the frozen retrieval rankings. [Exact fixture and measurements](../../results/nonlinear_retrieval_natural/timelineqa_counts_v1/) · [Earlier single-event probe](TEMPORAL_DATASETS.md#seeded-lifelogs-and-conflicting-provenance)
