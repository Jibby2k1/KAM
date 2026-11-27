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
