# Training for cutoffs between releases

Training on off-release cutoffs restored exact snapshot recovery on the original timestamps from two Eurostat indicators. With archive-shaped training and dot attention, unemployment recovery rose from 9.64% to 61.69% and the independent industrial-production panel from 5.41% to 93.06% at depth 48. Dallas release-day recovery moved from 54.88% to 50.20%. Mixed training showed a larger release-day trade-off.

The [cutoff-phase intervention](EUROSTAT_VINTAGES.md#cutoff-phase-intervention) showed that the original models relied on the latest snapshot falling at the query cutoff. This follow-up changes synthetic training only. Every synthetic archive episode samples a publication cutoff uniformly within the next discrete publication interval, then subtracts that common offset from all fact clocks. The interval is split into 64 positions, starting at zero and ending just before the next step. An originally available fact remains available, the next releases remain unavailable, and order at an address stays the same. The observations, entities, payloads, training steps, sequence lengths and per-seed batch samples stay fixed. Mixed training transforms only the archive-shaped half of the samples. A sham shifts only facts at a different address, leaving target release timing untouched.

The two Eurostat panels supply true calendar-day dates, the seasonal adjustment and unit are held fixed within indicator, and no labels from either panel enter training. Unemployment evaluates six countries from the original benchmark. Industrial production uses month-on-month percentages for total industry excluding construction from a separate official vintage table, with Austria, Belgium, Czechia, Denmark, Finland and Sweden. Its pinned response holds 19,074 published cells. Independent SQLite queries verify both cutoffs for 288 country-months, and 257 change numerical value between the cutoffs. Each Eurostat column comprises 3,456 paired lookup records across 576 questions, two recodings and three fits. Dallas contributes 3,504 records under its original release-day protocol.

| Training and geometry | Unemployment reference / sham / off-release | Industrial production | Dallas GDP |
| --- | ---: | ---: | ---: |
| Archive-shaped dot | 9.64 / 8.94 / 61.69 | 5.41 / 2.81 / 93.06 | 54.88 / 50.60 / 50.20 |
| Archive-shaped radial | 6.71 / 3.24 / 60.33 | 5.50 / 2.69 / 87.59 | 37.67 / 33.65 / 35.19 |
| Mixed dot | 3.01 / 3.24 / 22.66 | 0.20 / 0.20 / 83.54 | 57.08 / 58.68 / 23.23 |
| Mixed radial | 2.31 / 2.84 / 28.15 | 0.23 / 0.35 / 72.14 | 57.56 / 58.30 / 35.70 |

Values are percentages for exact key and snapshot at depth 48. For archive-shaped dot, the off-release effect versus reference was +52.05 percentage points [46.90, 57.38] on unemployment and +87.64 [82.93, 92.16] on industrial production. Dallas changed by −4.68 [−7.77, −1.45]. For mixed radial, off-release training lowered procedural correction accuracy from 82.81% to 70.70% at depth 48 and from 39.52% to 34.05% at depth 512. The same target timing augmentation improves external transfer but changes which timestamp profiles the model favors.

All control fits reproduced their archived loss samples, all held-out procedural decision flags, and the full unemployment readout. The study used the same tested cutoffs for comparisons. Later dates and mixtures of release phases need separate validation.

HPG L4 arrays 53002568 and 53002569. [Matched training and outcome records](../../results/nonlinear_retrieval_natural/clock_release_training_v3/) · [Industrial production fixture and source digest](../../results/nonlinear_retrieval_natural/eurostat_industry_v1/) · [Official industrial production vintages](https://ec.europa.eu/eurostat/databrowser/view/ei_is_m_vtg/default/table?lang=en)
