# Cross-domain transfer and controlled link noise

The frozen HotpotQA context-aware pair head transferred to some supplied Wikipedia pools. Its coverage features hurt MuSiQue under the same frozen weights. A plausible but irrelevant cross-reference can dominate both pair scoring and sequential retrieval. Training on such references recovered part of the 2Wiki loss on a fresh cohort, with no clear recovery on MuSiQue.

## Frozen transfer

All arms used the same frozen E5-small-v2 encoder, query adapter, FLAN-T5-small reader, and candidate pool for each question. The pair arms scored unordered pairs exhaustively. The sequential selector conditioned its second query on the first retrieved paragraph.

| Dataset | Questions | Pool contains both supports | Single-shot top two | Sequential | Original pair | Context-aware pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 768 | 768 | 333 / 143 | 447 / 165 | 520 / 175 | 570 / 186 |
| MuSiQue | 236 | 236 | 47 / 23 | 62 / 30 | 71 / 32 | 64 / 25 |
| News | 222 | 207 | 51 / 3 | 43 / 3 | 49 / 3 | 69 / 6 |

Cells are complete evidence selections and joint exact matches. The 2Wiki gain in joint exact match over sequential was 2.7 points with a paired 95% interval from 0.4 to 5.1 points. The 1.4-point gain over the original pair head had an interval from −0.4 to 3.3. On MuSiQue, the context head recovered seven fewer complete pairs and seven fewer joint answers than the original pair head. Its joint difference was −3.0 points with an interval from −6.8 to +0.8. [2Wiki results](../../results/nonlinear_retrieval_natural/dataset_breadth_v1/twowiki-summary.json) · [MuSiQue results](../../results/nonlinear_retrieval_natural/dataset_breadth_v1/musique-summary.json) · [News results](../../results/nonlinear_retrieval_natural/dataset_breadth_v1/news-summary.json)

The fresh 2Wiki and MuSiQue two-support fixtures excluded gold titles in the archived HotpotQA training exclusions and prior supporting-link fixture. MuSiQue also reserved questions used in earlier transfer runs. The news comparison used 222 questions with two distinct underlying articles, disjoint gold URLs across sampled questions, and a dense shortlist from 609 news articles. Articles were divided into 128-word windows with 32-word overlap, and the best scoring chunk represented each ranked article. The counts are specific to those restrictions.

## Feature ablation

Six heads used the same widths, initialization, minibatch order, 1,024 archived fitting pools, and full-pool objective. Feature groups were zeroed at fitting and evaluation. The full-feature refit reproduced every parameter of the original context head and every evaluated selection.

| Feature set | HotpotQA complete support | 2Wiki complete support | MuSiQue complete support | News complete support |
| --- | ---: | ---: | ---: | ---: |
| Full context | 346 | 570 | 64 | 69 |
| Without mention-local overlap | 347 | 564 | 63 | 66 |
| Without document-wide coverage | 210 | 510 | 73 | 47 |
| Without link or mention features | 279 | 399 | 43 | 77 |
| Original feature family with matched width | 212 | 514 | 70 | 47 |

Document-wide coverage explains the large HotpotQA gain. Mention-local overlap added little on these cohorts. Link indicators still supplied useful information on the Wikipedia datasets. The news sample had no linked gold pairs and the reader floor was low, so its selector differences carry little evidence about answer quality. [Matched refits and paired comparisons](../../results/nonlinear_retrieval_natural/link_feature_ablation_v1/summary.json)

## Controlled link noise

For each eligible question, selected a high-affinity distractor and one gold title such that their pair originally had no title link. Added an irrelevant cross-reference at the start of the distractor. The paired sham used the same syntax and normalized token count, with a fictitious reference. The physical text variants re-encoded only the edited passage. A graph-only control added exactly one directed edge while preserving original text, embeddings, and mention-local features. Both original supporting paragraphs stayed byte-for-byte unchanged.

| Dataset and method | Clean | Sham text | Linked text | Graph only |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki original pair, 768 | 520 | 540 | 95 | 267 |
| 2Wiki context-aware pair, 768 | 570 | 577 | 256 | 366 |
| 2Wiki sequential, 768 | 447 | 499 | 210 | 447 |
| MuSiQue original pair, 234 | 71 | 77 | 19 | 29 |
| MuSiQue context-aware pair, 234 | 63 | 64 | 32 | 41 |
| MuSiQue sequential, 234 | 62 | 70 | 37 | 62 |

The context head lost 41.8 points of complete evidence on 2Wiki from sham to linked text. The graph-only condition independently reduced complete evidence by 204 questions relative to clean. Sequential retrieval stayed identical under the graph intervention and fell under physical text injection. The injected reference can change both link features and the learned embedding of a wrong passage. The sham insertion itself changed rankings, making it an essential control. [Paired corruption audit](../../results/nonlinear_retrieval_natural/link_noise_v1/summary.json)

The follow-up held model size, optimizer, training examples and update count fixed. It compared repeated clean items, clean plus sham, and clean plus linked distractor training on the original 1,024 fitting questions. Four reference templates were used during fitting. Evaluation used the unseen wording “For further reading, consult” and independent question cohorts. On 2Wiki, linked-noise training recovered **357/768** complete pairs compared with **295/768** for sham training (**+8.1 points**, paired 95% interval **+5.9 to +10.3**). Clean support was 579 versus 573. On MuSiQue it recovered 33/191 versus 32/191 (**+0.5 points**, interval **−1.6 to +3.1**). Removing all explicit graph features under noise training reduced clean 2Wiki support to 371 and linked support to 218. The defense improved 2Wiki under the tested corruption while leaving a large residual weakness. [Independent robustness results](../../results/nonlinear_retrieval_natural/link_noise_training_v1/summary.json)

## Control for pool depth and distractor relevance

Both domains started from the supplied two gold paragraphs and eight supplied distractors. A shared bank of 8,192 outside passages provided nested expansions to 20 and 64 candidates, using either fixed seeded sampling or the highest frozen E5 affinity to the question. The gold paragraphs were unchanged and always present. Equal depth preserved the difference between domains.

| Candidate pool | 2Wiki sequential | 2Wiki context pair | 2Wiki without coverage | MuSiQue sequential | MuSiQue context pair | MuSiQue without coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supplied 10 | 447 | 570 | 510 | 90 | 85 | 93 |
| Random 64 | 427 | 560 | 507 | 88 | 84 | 91 |
| E5-hard 64 | 249 | 504 | 437 | 56 | 77 | 76 |

Counts are complete evidence out of 768 and 236. The coverage gain at hard depth 64 was +8.7 percentage points on 2Wiki (paired question-bootstrap interval +4.9 to +12.4) and +0.4 points on MuSiQue (−4.7 to +5.5). Random expansion had little effect relative to the hard expansion. The domain gap survived matched depth and this shared donor procedure. [Matched-pool results](../../results/nonlinear_retrieval_natural/matched_pools_v1/summary.json)

## More than two supports

Scored 192 fresh MuSiQue questions per hop count. The leaked-title guard excluded original HotpotQA training candidates and supporting-link study gold titles. Earlier MuSiQue question IDs were reserved. Some supporting titles recur across held-out questions. All retrieval arms got the true evidence-set size and the same total 384-token reader budget. The one-pass clique score summed frozen pair residuals over each candidate set. The tree variant used a maximum-scoring spanning tree.

| Hops | Independent top-k | Sequential | Original clique | Context clique | Context tree | Reader with gold evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 44 / 14 | 58 / 22 | 57 / 23 | 50 / 23 | 50 / 23 | 192 / 71 |
| 3 | 11 / 7 | 14 / 6 | 13 / 6 | 10 / 4 | 6 / 4 | 192 / 47 |
| 4 | 3 / 0 | 6 / 0 | 4 / 0 | 6 / 2 | 6 / 1 | 192 / 36 |

Each cell shows complete evidence and joint exact match. Gold link graphs were connected for 51, 12 and 5 questions across the three strata. Exact exhaustive set aggregation was feasible for the supplied pools of approximately 20 paragraphs, including 4,845 four-document sets at depth 20. Median routing time for the context clique stayed near 1.5 ms, while sequential routing rose from 3.3 to 9.2 ms across hop counts on the L4. These are resident candidate measurements through selection. The three- and four-hop arms recovered at most 14 and 6 complete sets, respectively. [Evidence-set and latency results](../../results/nonlinear_retrieval_natural/musique_multihop_v1/summary.json)

## Independent-domain reader controls

In news, all annotated fact spans were found in their source articles, but both exact spans were visible in the selected chunks for only 31 of 222 questions. The gold-fact prompt raised FLAN-T5-base exact match to 135/222 compared with 53/207 when the same reader saw the selected chunks from both gold articles. Question-only EM was 86/222 and is a strong confounder on that benchmark. This reader fixture supports a chunk-visibility diagnosis. [News matching and reader controls](../../results/nonlinear_retrieval_natural/dataset_breadth_v1/news-controls-summary.json)

QASPER provided a scholarly domain and independent human-annotated evidence alternatives. The selected 384 papers had 237 one-paragraph, 88 two-paragraph and 59 three-or-more minimum support requirements. In the initial run, complete evidence for two-paragraph questions was recovered zero or one times across all tested methods and oracle EM was only 10/384 overall. This fixture measures a weak reader and retriever under the short-context setup. Careful evidence windows and a reader suited to scholarly answers are prerequisites for using it to decide between retrieval methods. [QASPER initial results](../../results/nonlinear_retrieval_natural/qasper_evidence_v1/summary.json)

Using raw questions and raw annotated gold paragraphs increased small-reader EM to 27/384. The base reader reached 70/384 and token F1 of 0.414 with two raw gold paragraphs. Full gold paragraphs contained an extractive target string on 174 cases, with 152 still visible under the clipped evidence budget. Dense top-two complete evidence was 58 for title-prefixed text, 111 for raw text, and 85 for raw text with matched other-paper passages. This isolates input formatting and truncation as material experimental factors. [QASPER reader and format controls](../../results/nonlinear_retrieval_natural/qasper_evidence_v1/reader-controls-summary.json)

## Controlled reduction in lexical overlap

Built a retrieval probe from NoLiMa’s common direct and latent needles using two of its authored haystacks. It has 56 direct, 56 one-hop and 44 two-hop cases. Candidate pools grow from 16 to 64 to 256 passages. The lexical-noise arm uses the authors’ distractor strings. The goal is exact needle retrieval in a two-passage budget. The probe uses the same frozen QA reader and routing heads as the natural experiments, with empty graph features. The reported values are passage-retrieval probe scores.

| Needles | E5 selected needle at 16 | E5 selected needle at 64 | E5 selected needle at 256 | E5 selected lexical distractor at 64 | Gold-reader EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, 56 | 56 | 56 | 56 | — | 54 |
| One-hop, 56 | 19 | 9 | 7 | 47 | 30 |
| Two-hop, 44 | 13 | 6 | 1 | 37 | 21 |

On the distractor arm at depth 64, E5 needle retrieval fell from 9 to 6 and from 6 to 3 for one and two hops. The title-link-free context pair and sequential variants recovered at most four one-hop and three two-hop needles in that condition. The simple direct needle provides a high-recall control. The derived latent task gives a cheap, interpretable test of whether retrieval can associate a question with indirectly described evidence. More than one case can share a needle or haystack, so the table uses observed counts. [NoLiMa probe results](../../results/nonlinear_retrieval_natural/nolima_probe_v1/summary.json) · [Author paper](https://proceedings.mlr.press/v267/modarressi25a.html)
