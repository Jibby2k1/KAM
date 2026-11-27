# Nonlinear query and key projections in temporal memory

A nonlinear residual query and key map improved Eurostat generalization over an equal-size linear residual map with the same spherical score. The nonlinear-residual fit still fell below the original dot model on Dallas release-day data and deep synthetic corrections. The controlled comparison isolates the feature map under one fixed scale, update budget and source mixture.

All decoder variants use position-free two-layer context attention. The archive replay uses the same 3,200 updates and 10,556,800 tokens per seed and the same three held-out procedural draws and official vintage fixtures. Spherical scoring normalizes both query and key and uses a fixed scale of four. Both residual arms add two width-16 bottlenecks per layer, one to query and one to key, with zero-initialized output projections. The capacity control uses identity and the nonlinear arm uses SiLU. Initial logits matched exactly between both residual arms and the plain spherical control. The plain dot baseline reproduced its archived loss checks and held-out outputs.

| Model | Parameters | Median training seconds | Dallas strict at depth 48 | Unemployment strict at depth 48 | Industry strict at depth 48 | Corrections at depth 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dot | 94,528 | 13.92 | 57.08% | 3.01% | 0.20% | 30.92% |
| Fixed spherical | 94,528 | 15.72 | 17.27% | 1.22% | 0.49% | 0.59% |
| Fixed spherical + linear Q/K | 102,720 | 19.79 | 16.55% | 0.98% | 0.52% | 0.65% |
| Fixed spherical + nonlinear Q/K | 102,720 | 20.03 | 31.59% | 11.05% | 48.38% | 0.85% |

Nonlinear versus linear improved strict original unemployment by +10.07 points [8.04, 12.27] and industry by +47.86 [42.71, 52.78]. Later 2025 vintages moved from zero to 12.27% and 28.24% respectively, across just two quarters. This advantage appears at a matched geometry and parameter count. All spherical variants remained near one percent on depth-512 corrections. The independent mixed dot run with dates and address-local rank reached 99.02%. Rank supplies explicit within-address ordering and should be treated as an input-representation result. [Exact timestamp and rank comparison](ADDRESS_LOCAL_CLOCK.md)

HPG L4 array 53010260. [Matched seeds, provenance and archived numerical outputs](../../results/nonlinear_retrieval_natural/temporal_projection_control_v1/) · [Earlier short context geometry study](../nonlinear_retrieval/CONTEXT_CONFIRMATION.md)
