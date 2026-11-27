# Frozen HotpotQA pair scorer on native 2Wiki pools

At 128 candidates, the conditioned union raised support recovery from 320 to 383 of 1,024 and joint exact match from 149 to 165. The exact-evidence cohort showed a +2.34 percentage-point joint gain with a paired interval of +0.98 to +3.71. The annotation-incomplete cohort showed +0.78 points, with an interval of −0.20 to +1.95.

## Matched transfer

Used the sealed 2Wiki pools and the native HotpotQA encoder adapter, context heads, and fitted sentence scorer. Each scored pool used the same shortlist depth and frozen reader. The transferred policy received no 2Wiki training. The 64-candidate arm used the original dense prefix. The equal-depth control added the next 64 unconditioned nodes. The conditioned arm kept the same prefix and used the already sealed conditioned additions. Each support success requires both annotated article IDs. Joint exact match also requires an exact answer. [Sealed pools and evidence definitions](NATIVE_TWOWIKI.md)

| Cohort and metric | Dense 64 | Dense 128 | Conditioned 128 |
| --- | ---: | ---: | ---: |
| Exact evidence support, 512 | 180 | 182 | 222 |
| Exact evidence joint EM, 512 | 95 | 94 | 106 |
| Partial evidence support, 512 | 149 | 138 | 161 |
| Partial evidence joint EM, 512 | 57 | 55 | 59 |
| All support, 1,024 | 329 | 320 | 383 |
| All joint EM, 1,024 | 152 | 149 | 165 |

The equal-capacity joint lift was +1.56 points overall (paired interval +0.68 to +2.44), from 19 wins and three losses. Compared with the original dense 64, the conditioned policy gained 20 joint outcomes where that initial pool lacked a required title and lost seven among initially complete pools. For the exact-evidence cases whose original pool was incomplete, dense 128 and conditioned 128 recovered 7 and 47 support pairs, with joint counts of 3 and 15.

## Remaining errors

With conditioning, 198 questions lacked both required titles in the pool. Another 115 had both titles but no exact pair in the shortlist, 328 lost the pair during final ranking, and 218 selected the correct pair with an incorrect answer. The remaining 165 passed every stage. An outcome-aware oracle choosing among all three policies reached 172 joint exact matches and 403 support recoveries. It uses observed outcomes and is an upper bound on switching among these fixed outputs.

All four HPG L4 shards in array 52969603 replayed the same fixture and the published ranked pools. The run independently verified the fitted head and ranked-file fingerprints and loaded only the embeddings needed for each shard. [Ordered pair and reader outcomes](../../results/nonlinear_retrieval_natural/native_twowiki_rerank_v1/) · [Audit](../../scripts/analyze_kam_native_twowiki_rerank.py)
