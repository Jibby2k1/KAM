# Frozen clock transfer to monthly Eurostat releases

Unit-normalized clocks preserved every measured decision under three publication units. Lookup accuracy on the independent Eurostat release calendar remained low. At depth 48, radial episode-mean accuracy was 2.31% for the mixed fit that previously achieved 57.56% on Dallas GDP. Archive-shaped-only training raised the Eurostat number to 6.71% under the same clock. Both scores require the exact published snapshot and key.

The source is Eurostat's seasonally adjusted unemployment rate vintage table. The fixture pins the response containing 19,302 finite published cells from six countries. It tests every monthly reference observation in 2021–2024 for Germany, Spain, France, Italy, the Netherlands and Poland. Each question asks for the latest numerical entry published by either 90 or 365 days after the observation month ended. An independent SQLite selection reproduces all 576 lookup targets. At 174 of the 288 country-months the numerical value differs between those cutoffs. The median last-publication lag behind both requested cutoffs is 27 days. Sixty of the 576 queries land directly on a gold release day.

The 16-record bank keeps same-address snapshots immediately before and after both cutoffs and nearby distractors from the same and different countries. The 48-record bank retains the entire shorter bank and adds same-address snapshots and other distractors. The observation coordinate is measured in calendar months. Publication is supplied as days, days divided by 30, or days divided by 10. Any date recoding preserves snapshot order, cutoff eligibility, candidates, and ground truth. The protocol reruns the same three synthetic source fits per arm and compares them on the new fixture. No Eurostat observation enters training or selection of weights.

| Fit and geometry at depth 48 | Clock | Publication days ÷30 | Publication days ÷10 | Publication days |
| --- | --- | ---: | ---: | ---: |
| Mixed radial | Plain signed log | 3.59% | 3.79% | 3.94% |
| Mixed radial | Relative signed log, episode mean | 2.31% | 2.31% | 2.31% |
| Archive-shaped radial | Plain signed log | 7.55% | 3.39% | 1.39% |
| Archive-shaped radial | Relative signed log, episode mean | 6.71% | 6.71% | 6.71% |
| Archive-shaped dot | Plain signed log | 10.36% | 9.55% | 3.30% |
| Archive-shaped dot | Relative signed log, episode mean | 9.64% | 9.64% | 9.64% |

Both normalized clocks preserve all 3,456 outcome records per condition, including numerical exactness and unbound predictions. The mixed radial can produce a coincidentally correct number from another snapshot or key. The mean arm gets 35.45% numerical match versus 2.31% exact key and snapshot. The table should be read as an external failure of hard record selection under this temporal distribution, alongside successful invariance to changing the scale of a coordinate. Intervals resample calendar quarters jointly across all six countries and fit seeds. The Dallas panel uses actual archive release dates as cutoffs. Eurostat tests fixed offsets from month end. A matched cutoff-phase intervention can distinguish a learned readiness boundary from other differences in candidate topology.

HPG L4 arrays 52999598–52999599. [Pinned source response, manifest, numerical outcomes and paired estimates](../../results/nonlinear_retrieval_natural/eurostat_vintages_v1/) · [Eurostat rate vintages](https://ec.europa.eu/eurostat/databrowser/view/ei_lm_m_vtg/default/table?lang=en) · [Vintage methodology](https://ec.europa.eu/eurostat/cache/metadata/en/euroind_vtg_esms.htm)
