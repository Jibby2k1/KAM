# Repeated effective dates in long revision histories

Replacing stale and late records with label-preserving same-effective corrections caused both dot and radial decoders to fail on deep temporal retrieval. Training entirely on the corrected sequences did not recover the ordinary long-history performance or transfer to held-out workbook revisions.

The intervention kept the same query, record IDs, identities, values, order, depth mix and batch draws. A stale, same-key record moved to the target's effective date and was assigned a recorded date earlier than the target. A same-key late record moved to that effective date and retained its post-cutoff recorded date. The independent typed exact solver selected the original alias and value on all modified histories. This changes which clock separates the eligible records. Both training arms started from identical shared parameters per seed and used eager FP32 for 9,600 updates, 39,307,200 tokens, and the same short-first order. The primary synthetic panels contain 384 other examples per seed and depth.

| Geometry and history used to fit | Ordinary held-out, depth 1,024 | Corrected held-out, depth 1,024 | Earlier workbook, depth 48 | Later workbook, depth 48 |
| --- | ---: | ---: | ---: | ---: |
| Dot, original | 36.72% | 0.43% | 0.00% | 0.00% |
| Dot, corrected | 0.35% | 0.69% | 0.73% | 0.26% |
| Radial, original | 50.26% | 0.69% | 0.00% | 0.00% |
| Radial, corrected | 1.04% | 1.04% | 0.36% | 0.26% |

Each cell is mean joint address-and-record accuracy across three matched seeds. The separate archived panels use 160 earlier and 32 later observation groups, two cutoffs, and two symbol recodings. They exercise real release clusters with independent source tables. The result isolates a difficult training distribution formed by changing clock coordinates, and documents failure for both tested geometries and this optimizer. Earlier archive-shaped training supplied different revision, timing and payload configurations and performed much better. Those results apply to the archive-shaped training distribution.

The first HPG array, 52983793, finished one seed within its requested eight minutes. The same sealed inputs for the remaining seeds finished in array 52985015 with a larger time limit. [Paired seed and archive outcomes](../../results/nonlinear_retrieval_natural/temporal_correction_curriculum_v1/) · [Earlier archive-topology comparisons](BITEMPORAL_MODEL.md#transfer-to-actual-revision-sequences)
