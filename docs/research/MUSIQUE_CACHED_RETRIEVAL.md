# Cached first-hop retrieval on native MuSiQue pools

Use new two-hop questions and supporting titles from the published MuSiQue training and development splits. Reserve question IDs from both earlier paired endpoint studies and the separate title-field study. Reserve normalized wording and all supporting titles from both earlier paired studies. Relation templates can recur and are identified in the fixture. Each first-hop pool contains the annotated earlier passage, the alternate earlier passage from the question's matched foil, and distinct nonsupporting paragraphs from the question's own MuSiQue pool. Give all retrievers the same candidates. Supply the two matched endpoints to Llama and vary the first-hop pool and endpoint orders independently. Qualify visible evidence using the pinned Llama tokenizer before scoring. [MuSiQue source](https://github.com/StonyBrookNLP/musique) · [Frozen manifest](../../results/general_benchmarks_musique_cached/manifest.json)

Compare BM25 with cached document statistics, frozen E5-small-v2 with cached normalized passage embeddings, and equal-weight reciprocal rank fusion of both rankings with constant 60. BM25 reads full passages and E5 uses a 256-token cap. Dense and hybrid perform live encoding of each question. All three show their top-ranked earlier passage to the same single-view Llama endpoint reader. Use its A/B-only vocabulary projection, with the same reader prompt and 512-token passage cap in all routes. Freeze Llama 3 8B Instruct, the three KAM and dot seeds at KL 0.5, and untouched Llama. [Pinned E5-small-v2](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52) · [Prior comparison and projection audit](OVERNIGHT_ROUTING.md)

Report first-hop recall at one and three across both first-hop pool orders. Joint recovery requires the annotated first hop and correct endpoint for all four presentations. The primary deadline metric also requires each presentation's median of three measurements to be at most 60 ms. Interleave route order on warmed RTX PRO 6000 GPUs. Timing includes live retrieval, query encoding where applicable, reader prompt construction and tokenization, GPU transfer and endpoint choice. Passage clipping, index construction, passage embedding, model loading and external serving remain outside the timed region. Report publisher splits separately and compute paired relation-family intervals conditional on the frozen checkpoints. This is fixed-pool joint passage selection with a supplied endpoint pair.

## Results

The tokenizer visibility check retained 88 questions in 22 families from the publisher's training split and 12 questions in four families from development. Every first-hop pool contained 20 passages. All question IDs and supporting titles were new relative to the earlier paired studies, and the relation templates recur. The sampled questions and paired supporting passages had zero exact normalized-text matches in the frozen adapter-training sources. The offline cache encoded 1,665 distinct passages once. [Cache record](../../results/general_benchmarks_musique_cached/cache.json) · [Source audit](../../results/general_benchmarks_musique_cached/source-audit.json)

| Training split, 88 questions | First-hop R@1 | First-hop R@3 | KAM joint ≤60 ms | Dot joint ≤60 ms | Untouched Llama joint ≤60 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 69.32% (61/88) | 82.95% | 44.70% | 52.65% | 30.68% |
| Cached dense | 88.64% (78/88) | 95.45% | 52.27% | 64.02% | 36.36% |
| Hybrid | 81.82% (72/88) | 97.73% | 50.00% | 60.61% | 34.09% |

First-hop recall was identical across the seven readers. All joint successes met the 60 ms budget. Cached dense improved over BM25 by +7.58 percentage points for KAM [−0.44, +14.58], +11.36 for dot [+1.32, +20.00], and +5.68 for untouched Llama [0.00, +11.63]. With the same dense retrieval, KAM trailed dot by 11.74 points [−18.38, −4.58] and led untouched Llama by 15.91 [+9.52, +23.40]. These 95% intervals bootstrap question families conditional on the evaluated checkpoints. Hybrid offered the best top-three coverage. Cached dense gave the strongest top-one route for each reader.

| Training-split latency | Retrieval p50 across readers | KAM p50 / p95 | Dot p50 / p95 | Untouched p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.04 ms | 28.52 / 34.77 ms | 28.11 / 34.39 ms | 27.77 / 34.07 ms |
| Cached dense | 2.01–2.04 ms | 30.47 / 36.93 ms | 29.99 / 36.44 ms | 29.67 / 36.04 ms |
| Hybrid | 2.06–2.08 ms | 30.58 / 36.97 ms | 30.15 / 36.49 ms | 29.78 / 36.08 ms |

| Development split, 12 questions | First-hop R@1 | First-hop R@3 | KAM joint ≤60 ms | Dot joint ≤60 ms | Untouched Llama joint ≤60 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 50.00% (6/12) | 83.33% | 50.00% | 44.44% | 33.33% |
| Cached dense | 83.33% (10/12) | 91.67% | 83.33% | 77.78% | 66.67% |
| Hybrid | 75.00% (9/12) | 91.67% | 75.00% | 69.44% | 58.33% |

Dense led on the development split too. KAM and dot reversed order there, on just 12 questions in four families. The smaller stratum is a replication of the retrieval ordering with limited information about the adapter ordering. All seven matched HPG jobs completed. [Paired summary](../../results/general_benchmarks_musique_cached/summary.json) · [Per-presentation outcomes](../../results/general_benchmarks_musique_cached/outcomes.json.gz) · [HPG receipts](../../results/general_benchmarks_musique_cached/scheduler.tsv)
