# Hard endpoint selection with two plausible MuSiQue passages

On a fresh paired two-hop panel, using a verified earlier passage to choose one of two endpoints improved E5 selection from 252 to 328 of 336 presentations. With FLAN-T5-large, generating from just the selected endpoint improved joint exact match from 181 to 224, compared with reading both endpoints and the earlier evidence together. At width eight on HPG L4, the paired median wall-time saving was 224 ms per batch, including live E5 query encoding and reader generation.

## Cohort and controls

Two different native MuSiQue development questions share a final relation template. Their correct endpoint passages have distinct titles, verified bound entities and accepted literal answers. Each early context must mention the corresponding bound entity. The original question cannot contain that entity or an accepted answer, and neither endpoint contains the competing question's bound entity or answer. Candidate and true earlier evidence roles use the author annotations. Previous fixture IDs and duplicate normalized question wordings are excluded. Evidence titles do not repeat across selected pairs.

The offline pairing filters produced 97 two-hop and one four-hop pairs. After verifying that the fixed passage windows retained both bindings and endpoint answers, 84 two-hop pairs and one four-hop pair remained. The main result covers 168 questions from 35 relation families, each presented in both endpoint orders. The four-hop pair is recorded descriptively. The matched true and cross-paired prefixes have exactly the same token count. Choosing a pair with these filters creates a targeted challenge for endpoint competition.

The single-shot E5 control uses the original question and both candidate embeddings. Conditioned E5 also sees the correct earlier paragraph text, without the author answer label. The cross-prefix control supplies the competing matched paragraph of equal length. The bound oracle uses the author final template and true earlier answer. The reader always generates a free-form short answer without a supplied answer list.

## Accuracy at matched candidates

| E5 scoring input | Correct endpoints / 336 | Both questions correct in both orders / 84 |
| --- | ---: | ---: |
| Original question | 252 | 45 |
| Original question and verified earlier text | 328 | 80 |
| Original question and cross-paired earlier text | 44 | 1 |
| Author final subquestion and true binding | 334 | 83 |

E5 returned the same endpoint under both orders for all 168 questions. Adding true earlier evidence gained 22.62 points relative to the original-question E5 baseline [12.81, 30.56], with intervals grouped by the 35 relation families.

Joint correctness below requires the correct endpoint and exact normalized answer. For the broad two-endpoint reader with true earlier text, both required passages are already present. On all hard-route arms other than the endpoint-only view, the reader still gets the **true** earlier paragraph. The cross-prefix arm changes the selector's input only.

| Reader input and endpoint policy | Base joint / 336 | Large joint / 336 | Large foil answers |
| --- | ---: | ---: | ---: |
| True earlier text and both endpoints | 159 | 181 | 17 |
| True earlier text and original-question E5 choice | 144 | 144 | 38 |
| True earlier text and conditioned E5 choice | 200 | 200 | 4 |
| True earlier text, conditioned choice, rival masked at matched physical length | 197 | 206 | 4 |
| Conditioned E5 choice alone | 210 | 224 | 4 |
| Author endpoint alone | 214 | 228 | 0 |
| True earlier text and cross-prefix E5 choice | 26 | 26 | 88 |

The large endpoint-only hard route gained 12.80 points [5.65, 19.37] over the two-endpoint reader, with 60 paired gains and 17 losses. The number of questions correct in both endpoint orders rose from 80 to 112. The same-width mask improved outcomes, showing an effect from which text remains visible. The narrower reader prompt reduced median live tokens from 289 to 122. Selection consumes verified earlier evidence. The generation step can then focus on the selected endpoint. Finding that earlier paragraph from a corpus remains outside this comparison.

## Warm paired latency

Generation timings use the identical frozen quality panel and preloaded candidate embeddings. The two-passage baseline starts from pretokenized inputs. The hard route includes E5 query tokenization, encoding, scoring and prompt assembly before generation. Timings are synchronized and exclude initial passage retrieval, corpus embedding and model loading. The first batch for each reader, shard and width is excluded. Every timed exactness outcome matched its frozen quality arm.

| Reader and batch width | Paired batches | Read both, median | Route and read one, median | Median paired saving |
| --- | ---: | ---: | ---: | ---: |
| Base 1 | 44 | 32.85 ms | 32.60 ms | −0.05 ms |
| Base 4 | 37 | 89.10 ms | 62.16 ms | 24.78 ms |
| Base 8 | 15 | 181.36 ms | 104.76 ms | 73.16 ms |
| Large 1 | 44 | 82.73 ms | 67.13 ms | 14.22 ms |
| Large 4 | 37 | 268.79 ms | 171.46 ms | 95.78 ms |
| Large 8 | 15 | 536.82 ms | 300.37 ms | 224.35 ms |

HPG L4 arrays 53016711, 53017541 and 53018115. [Sealed panel manifest, matched outcomes and timings](../../results/nonlinear_retrieval_natural/musique_endpoint_routing_v1/) · [Four-hop terminal evidence controls](MUSIQUE_TERMINAL_EVIDENCE.md)

## Retrieving the earlier paragraph from the native corpus

Opening the first hop preserved most of the conditional routing gain. On one canonical ordering of the 168 two-hop questions, dense top-one selected the annotated earlier paragraph on 141. It then chose the correct endpoint on 160, and 109 also had exact answers. Ninety-three of those 109 matched the full annotated chain. The corresponding original-question route gave 72 exact endpoint-and-answer pairs, and providing the verified earlier paragraph gave 112.

The first step queries the same native bank of 101,748 paragraphs with frozen E5, BM25 or rank fusion. It excludes the two visible endpoint titles and uses the original question without author intermediate answers. For each eligible earlier candidate, the frozen selector reads up to 128 tokens of its text. The primary route uses top one. A four-candidate control chooses the candidate with the largest absolute E5 margin between the two endpoints. Both routes feed the same reader the selected endpoint alone. This experiment opens the earlier search and keeps endpoint capacity and membership fixed.

| Earlier search and selection | Annotated earlier passage chosen | Correct endpoint | Joint endpoint and answer | Full annotated chain and answer |
| --- | ---: | ---: | ---: | ---: |
| Dense, top one | 141 | 160 | 109 | 93 |
| Dense, margin over top four | 128 | 155 | 105 | 86 |
| BM25, top one | 118 | 150 | 102 | 79 |
| BM25, margin over top four | 112 | 153 | 105 | 76 |
| Hybrid, top one | 136 | 158 | 107 | 89 |
| Hybrid, margin over top four | 124 | 157 | 107 | 84 |

Dense top-one gained 22.02 points [8.16, 31.47] over original-question endpoint scoring and was three questions below the true-earlier control [−4.76, 0.00 points]. The margin-selected beam had more earlier passages available and chose the annotated one less often. A complete open-corpus comparison would also generate the endpoint candidates and evaluate reachability.

HPG L4 array 53019537. [Corpus checksum, paired first-hop and complete-chain outcomes](../../results/nonlinear_retrieval_natural/musique_endpoint_open_v1/)

## Searching for both evidence passages

Conditioned corpus search also helped when both evidence passages had to be retrieved from the native corpus. Dense search with one conditioned hop selected both annotated passages for 60 of 168 questions, compared with 25 for same-pool sequential ranking and eight from the direct top two. Reading the second selected passage gave 43 joint answers, versus 17 and six. The candidate budget was 128 in every arm.

The initial BM25, dense or hybrid ranking is fully open. No annotated premise, endpoint candidates, intermediate answer or final answer is used in search. The direct route takes its top two. The sequential route fixes the same leading passage and uses that passage to reformulate the query. The same-pool sequential arm selects the second document within the initial top 128. The corpus sequential arm keeps the first 64 and adds the first 64 distinct new paragraphs from the conditioned rank. Every arm uses the same FLAN-T5-large reader and per-passage limit of 160 tokens. It either reads the two selected passages or only the second one. Joint exact requires both selected IDs and an exact answer in either readout.

| Search | Both supports available | Both selected | Joint, both read | Joint, second read |
| --- | ---: | ---: | ---: | ---: |
| Dense, direct | 56 | 8 | 4 | 6 |
| Dense, same-pool sequential | 56 | 25 | 15 | 17 |
| Dense, conditioned corpus search | 128 | 60 | 35 | 43 |
| Hybrid, direct | 54 | 8 | 3 | 5 |
| Hybrid, same-pool sequential | 54 | 27 | 16 | 17 |
| Hybrid, conditioned corpus search | 123 | 60 | 36 | 43 |
| BM25, direct | 39 | 5 | 3 | 2 |
| BM25, same-pool sequential | 39 | 18 | 11 | 11 |
| BM25, conditioned corpus search | 115 | 50 | 31 | 35 |

On dense search with the identical second-only readout, conditioned corpus retrieval gained 15.48 points over same-pool sequential [5.33, 24.44], with 28 gains and two losses. Reading just the second selected passage increased the dense conditioned count from 35 to 43 [0.00, 10.32 points]. The fixed pair score of 109 endpoint-and-answer matches still required the small two-endpoint candidate set. Here both retrieval stages are open. The cohort remains filtered for clear bound entities and plausible competing endpoints, so this comparison measures transfer within that controlled slice.

The dense conditioned route partitions the 168 cases into 40 incomplete pools, 11 complete pools with a nonannotated anchor, 57 complete pools with an annotated anchor and a missed second passage, 17 selected pairs with an inexact terminal readout, and 43 joint successes. Among the 68 complete pools that lose a support at selection, 57 already have a usable anchor. That makes ranking the second passage the largest measured selection target on this slice.

HPG L4 array 53020278. [Fully open candidate lists and paired results](../../results/nonlinear_retrieval_natural/musique_open_both_v1/)

## Reranking the frozen second-passage choices

Giving priority to candidate titles explicitly mentioned in the first passage increased the dense conditioned pair count from 60 to 70 and the joint count from 43 to 52. This exploratory check follows the same cohort's error audit. It freezes all 128 available candidates, the selected anchor, and the terminal-only reader for every arm. The conditioned reference reproduced both the original selected documents and all 168 answer flags.

A title mention is a normalized token-phrase match of at least two words within the first passage's encoded 128-token window. The hard preference selects the highest conditioned score among mentioned candidates when any are present. A shortlist preference only considers mentions among the top eight. The question blend adds one quarter original-query score. The anchor score adjustment subtracts one quarter of the candidate's cosine similarity to the first passage, testing whether redundant content dominates the conditioned score.

| Frozen second-passage policy | Both supports selected | Joint exact match | Selected titles mentioned |
| --- | ---: | ---: | ---: |
| Conditioned E5 | 60 | 43 | 48 |
| Favor mentioned titles | 70 | 52 | 76 |
| Favor mentions among top eight | 68 | 50 | 67 |
| Subtract anchor similarity | 68 | 49 | 51 |
| Blend original question | 46 | 33 | 43 |

The unrestricted mention preference gained ten joint answers and lost one, a paired 5.36 percentage points [0.74, 9.68]. The shortlist preference gained seven without a loss on this cohort. Those intervals resample relation families and describe this diagnostic panel. Fresh coverage and naturally linked distractors should guide whether to adopt the preference elsewhere.

HPG L4 array 53021468. [Case-level ranking and recovery](../../results/nonlinear_retrieval_natural/musique_open_rerank_v1/)
