# Executable object memory in BABILong

A typed chronological replay matched all 1,200 official answers across four BABILong tasks and three context lengths. When the same replay could inspect only eight retrieved windows, joint recovery fell sharply for object location and preceding location. The inventory questions needed fewer distinct facts and were easier to recover.

The pinned benchmark mixes synthetic event facts with PG-19 book text. The audit parses moves, pickups, drops and transfers, tracking object ownership and location in source order. It compares answers with the official fields and verifies a set of trace witnesses for each nonempty derivation. At the author's 0k, 16k and 64k lengths, 386 episodes preserve both question and extracted event trace row by row. QA3 contributes 86 of its 100 episodes to that matched comparison. All 1,200 official rows contribute to the executable audit. Trace witnesses come from the independent replay rather than author passage annotations.

The retrieval benchmark uses 80-word windows with stride 60, with a capacity of eight windows for every arm in the table. BM25 and E5 have access to the entire book and inserted events. Sequential E5 starts with four windows, infers an actor from the most recent object action it found there, and uses actor-conditioned search to fill the remaining four slots. The grammar-filtered arm admits only windows that contain a recognized event and then ranks by the same dense scores. All methods pass selected windows to the same chronological replay.

| Task | Episodes | Dense joint, 16k / 64k | BM25 joint, 16k / 64k | Sequential joint, 16k / 64k | Grammar-filtered dense joint, 16k / 64k |
| --- | ---: | ---: | ---: | ---: | ---: |
| QA2, object location | 100 | 4 / 1 | 7 / 2 | 29 / 13 | 43 / 41 |
| QA3, preceding location | 86 | 1 / 0 | 2 / 0 | 4 / 1 | 6 / 3 |
| QA7, count carried objects | 100 | 67 / 45 | 49 / 32 | — | 91 / 91 |
| QA8, list carried objects | 100 | 77 / 64 | 76 / 67 | — | 96 / 95 |

Joint requires an exact denotation and recovery of a nonempty witness. Selective replay can agree with an empty inventory through absence of evidence, so the summary separates empty and nonempty gold answers. QA7 nonempty dense joint fell from 37 to 29 of 54 and grammar-filtered dense held at 52. QA8 nonempty dense joint fell from 50 to 39 of 66 and grammar-filtered dense reached 64 and 63. The longer official panels preserve matched event traces and can use different sampled book context. Median candidate windows grew from about 182–186 to 735–751. For QA3 the number of event-containing windows grew from 45 to 51 at the median, leaving a substantial retrieval burden even after schema filtering.

HPG arrays 52991336 and 52995611 measured both context lengths on L4. The additional B200 QA2 replay, 52991337 shard four, reproduced every numerical ranking record on all 100 questions. A task-specific grammar is a high-information control on this benchmark. Retrieval on arbitrary prose needs evidence linking and source interpretation beyond the recognized event templates.

## Valid event language in untrusted filler

A second audit inserted an independently sampled, quote-like block of well-formed event sentences around each authoritative trace. It used eight random snippets per episode and placed the identical snippet before or after the true trace. The target and provenance stayed fixed. Treating every well-formed sentence as an update produced the following exact counts.

| Task | Trials | Snippet before trace | Snippet after trace | Paired change, percentage points |
| --- | ---: | ---: | ---: | ---: |
| QA2 | 800 | 800 | 527 | −34.13 [−37.63, −30.75] |
| QA3 | 688 | 688 | 636 | −7.56 [−9.74, −5.52] |
| QA7 | 800 | 713 | 524 | −23.63 [−27.50, −19.75] |
| QA8 | 800 | 698 | 507 | −23.88 [−27.13, −20.88] |

Clearing state updates with a trusted source flag restored all 3,088 exact outcomes at each placement. Resampling clustered the eight draws from each original episode. This is a simulated case where quoted content and executable memory writes share a grammar. The original official retrieval comparison above uses natural book filler. [Paired provenance replay](../../results/nonlinear_retrieval_natural/babilong_replay_v1/provenance/)

[Audited manifests, ranks and comparisons](../../results/nonlinear_retrieval_natural/babilong_replay_v1/) · [Official benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c0d62e70dbc659cc9bd44cbcf1cb652f-Abstract.html) · [Pinned data](https://huggingface.co/datasets/RMT-team/babilong/tree/ee0d588794c7ac098062ee0d247c733d62e94fe2)
