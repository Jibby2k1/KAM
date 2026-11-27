# Initial indexing precision and shortlist-only text normalization

Full half-precision scan followed by FP32 rescoring of its shortlist cut initial HotpotQA retrieval from 32.44 to 15.99 ms at the median in a 192-request HPG L4 replay with the fused second hop. Median request latency was 108.13 ms with FP32 initial search and 92.51 ms with rescoring. Both arms recovered 97 complete support pairs and 57 joint exact matches.

## Paired initial search

The comparison uses the same 192 fixture questions within each arm, randomizes policy execution per question, and uses the same warmup, corpus, second-hop search, ranker, and fixed reader. Half-precision scan retains 96 preliminary entries for FP32 rescoring. Using a full scan before rescoring is an empirical precision control. The rescored initial top-64 set matched on all 192 requests and the order matched on 191. The one ordering change left the anchor, conditioned candidates, chosen pair and outcome identical.

| Fixed second-hop search | Initial arm | Initial p50 (ms) | Request p50 (ms) | Request p95 (ms) | Paired median request delta (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Full scan and FP32 rescore | FP32 full scan | 32.43 | 122.40 | 171.29 | |
| Full scan and FP32 rescore | Half scan and FP32 rescore | 15.98 | 106.31 | 154.24 | −16.23 [−16.39, −16.10] |
| Fused 256-list search | FP32 full scan | 32.44 | 108.13 | 157.06 | |
| Fused 256-list search | Half scan and FP32 rescore | 15.99 | 92.51 | 141.30 | −16.13 [−16.24, −15.93] |

The interval is a paired bootstrap over requests. The two second-hop configurations run as independent paired comparisons. Their baseline timings may differ from the earlier four-arm replay.

## Normalizing only shortlisted text

HPG array 52981212 independently replayed the same 192 requests with the rescored initial search and fused 256-list second hop. Pair scoring and shortlist formation were unchanged. The alternate path normalizes the sentence text only for documents in proposed pairs, with reuse across the broad and conditioned rankings within a request. This reduced normalized documents from 33,854 to 6,331. Rankings and reader records matched on all 192 requests.

Median remaining-rerank time was 50.41 versus 48.74 ms, with median paired change −1.29 ms [−1.46, −1.15]. Median end-to-end latency was 95.53 versus 94.28 ms, with median paired change −1.45 ms [−1.66, −1.08]. The sentence work saved was a small part of reranking time. Inspect pair-feature construction and sentence scoring separately before optimizing the largest stage.

[Measured runs and summaries](../../results/nonlinear_retrieval_natural/fullwiki_coarse_v1/) · [Initial analysis](../../scripts/analyze_kam_fullwiki_rescored_initial.py) · [Text preparation analysis](../../scripts/analyze_kam_shortlist_text.py)
