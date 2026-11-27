# Learned event pointers on ordered schedules

The trained dot and radial pointers recovered 52.77% and 49.53% of complete original-and-intervention pairs after receiving an ordered list of durations and a typed anchor. The candidate-local model recovered 1.08%. The context pointers generalized to held-out plans of comparable length and had low accuracy at 64 and 128 events.

## Matched training

Three L4 seeds each fit three small models for 2,400 optimizer steps and 2,572,800 tokens per model. The training data were independently generated duration arrays at 16 and 32 events, with a randomly anchored elapsed time and a target event. All arms received the same normalized local duration, anchor-relative position, elapsed time, anchor marker, sequence length, and mode marker. The 75,649-parameter local MLP scored each event independently. The dot and radial models used two KAM context layers and a pointer readout, with 79,552 and 79,688 parameters. A mean-duration rule and exact cumulative prefix were fixed controls. [Runner](../../scripts/run_kam_duration_pointer.py) · [Raw records](../../results/nonlinear_retrieval_natural/duration_pointer_v1/)

| Held-out events and max duration | Local | Dot context | Radial context | Mean duration | Exact prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 90 | 45.31% | 86.33% | 80.60% | 45.25% | 100% |
| 32 / 90 | 35.35% | 80.08% | 80.60% | 36.59% | 100% |
| 64 / 90 | 22.53% | 10.22% | 8.59% | 26.95% | 100% |
| 128 / 90 | 13.93% | 1.76% | 1.37% | 21.09% | 100% |
| 64 / 45 | 9.64% | 6.90% | 6.84% | 27.08% | 100% |
| 128 / 22 | 4.69% | 2.08% | 1.50% | 22.01% | 100% |

Values are means across three independent 512-question test panels. The 64/45 and 128/22 panels keep expected total duration near the 32-event training scale and change the individual duration distribution. Accuracy at greater depths stayed low under those changes.

## Transfer to matched schedule edits

At evaluation, a deterministic parser supplied the named anchor and elapsed time from the authored question. The original, sham, and intervention forms shared the same event order and duration multiset. These inputs test numeric event addressing inside the authored plans. They omit natural question parsing from the learned task. [Duration swaps](TEMPORAL_DATASETS.md#duration-changes-with-the-same-numbers)

| Selector | Original correct | Sham correct | Intervention correct | Both original and intervention | Prediction changed on sham |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local | 12.48% | 12.53% | 16.60% | 1.08% | 5.83% |
| Dot context | 71.78% | 71.58% | 71.90% | 52.77% | 0.97% |
| Radial context | 69.03% | 68.77% | 69.65% | 49.53% | 1.82% |
| Mean duration | 29.07% | 29.07% | 25.06% | 0.00% | 0.00% |
| Exact prefix | 100% | 100% | 100% | 100% | 0.00% |

Each seed saw the same 1,173 archive-derived questions from ten plans. The dot and radial selections changed on 73.77% and 73.15% of label-changing interventions. The summary provides seed and plan counts so that question reuse stays visible. HPG array 52964534 produced the training runs. [Numerical audit](../../scripts/analyze_kam_duration_pointer.py) · [Exact execution timing](SERIAL_DURATION_INDEX.md)
