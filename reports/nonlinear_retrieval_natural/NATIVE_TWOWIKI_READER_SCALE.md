# Reader capacity and precision on fixed 2Wiki evidence

A larger frozen reader did not improve joint exact match on the retrieved 2Wiki article pairs. It answered more questions with gold evidence prompts, especially when the supporting sentences were supplied. On L4, BF16 roughly halved the base reader's batched generation time with small changes in exact match.

## Matched result

Each reader saw the same 3,148 unique prompt strings in the same order and with identical token IDs after truncation. The four prompt types used the frozen HotpotQA scorer's chosen pair, the in-domain 2Wiki scorer's chosen pair, gold native-article leads, or gold annotated sentences. The selected article IDs and order were fixed before changing readers. The small replay reproduced all 4,096 archived exactness labels. Decoding used the same greedy generation configuration and 20-token output cap. [Frozen refit](NATIVE_TWOWIKI_REFIT.md) · [Matched gold evidence](NATIVE_TWOWIKI.md)

| Prompt source, 1,024 questions | Gold pair in prompt | Small FP32 answer EM | Base FP32 answer EM | Base BF16 answer EM | Joint EM, small / base FP32 / base BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen pair | 383 | 238 | 226 | 230 | 165 / 160 / 162 |
| Refit pair | 425 | 242 | 235 | 237 | 162 / 158 / 160 |
| Gold articles, native leads | 1,024 | 320 | 333 | 334 | 320 / 333 / 334 |
| Gold annotated sentences | 1,024 | 407 | 431 | 434 | 407 / 431 / 434 |

For the refit pair, base BF16 changed joint EM by −0.20 percentage points versus small (paired 95% interval −2.34 to +1.95). In the gold-annotation control it changed EM by +2.64 points (−0.59 to +5.86). On the 86 questions whose gold pair was recovered only by the refit, the selected-evidence reader answered 15 with small and 17 with either base precision. Both base modes answered 33 of those questions when given the annotated sentences. These results support testing how to expose better evidence to the reader before another increase in model size.

## Matched L4 generation

Each shard independently warmed a 32-prompt batch before timing all its unique prompts. Median times across shards were 4.50 seconds for small FP32, 13.82 seconds for base FP32, and 6.81 seconds for base BF16. These are batched reader measurements. They exclude retrieval, reranking, and queueing. Base BF16 was about 2.03 times faster than base FP32 on this workload. It changed joint EM by two questions relative to base FP32 on each selected-pair route and by three in the gold-annotation control.

HPG arrays 52975258, 52975259, and 52975260 each ran four L4 shards. [Pinned FLAN-T5-base checkpoint](https://huggingface.co/google/flan-t5-base/tree/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2) · [Paired metrics and replay](../../results/nonlinear_retrieval_natural/native_twowiki_reader_scale_v1/) · [Runner](../../scripts/run_kam_native_twowiki_reader_scale.py)
