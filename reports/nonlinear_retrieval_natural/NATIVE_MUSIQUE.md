# MuSiQue retrieval with complete paragraph evidence

On 576 fresh MuSiQue-Ans questions, conditioned expansion from two dense anchors recovered every required paragraph on 212 questions, compared with 176 for unconditioned dense search at the same capacity of 128. The paired gain was 6.25 percentage points, with a shared-evidence-cluster interval of +4.03 to +8.93. The gains came from two- and three-hop questions. Four-hop full recall remained low.

## Corpus and fixed evaluation

Built a bank of 101,748 unique paragraphs from 447,039 occurrences in the released train and development contexts. Deduplication used normalized title and full paragraph text. This is a source-specific paragraph retrieval setting. Every test annotation has an exact paragraph ID in the bank. The 576 questions exclude 1,319 previously used IDs and wordings and share 940 distinct required paragraphs. The corpus contains the development paragraphs, and the frozen encoders and expansion rules received no fit examples from this cohort. [MuSiQue source](https://aclanthology.org/2022.tacl-1.31/) · [Earlier lead-index limitation](CROSS_DATASET_WIKIPEDIA.md)

| Required chain recovered at 128 | Two hops, 224 | Three hops, 192 | Four hops, 160 | All, 576 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 112 | 34 | 7 | 153 |
| Dense | 130 | 39 | 7 | 176 |
| Hybrid reciprocal rank fusion | 135 | 56 | 12 | 203 |
| Dense 64 plus one conditioned branch | 144 | 58 | 6 | 208 |
| Dense 64 plus two conditioned branches | 146 | 59 | 7 | 212 |

The conditioned arms preserve the same dense top 64. Each branch re-encodes the question with one of the two initial dense hits and blends that query with the original. The two-branch version interleaves candidates until it has the same 64 unique additions as the one-branch version. Both use more query work than the unconditioned controls. No downstream reader outcomes are measured here.

The two-branch gain over the one-branch policy was four questions, with a cluster interval of −1.06 to +2.26 points. For four-hop questions, hybrid recovered 332 of the 640 gold paragraph occurrences and two-branch conditioning recovered 268. Hybrid's four-hop full-chain advantage over dense was five questions, with an interval spanning zero. Resampling uses connected components of questions that share at least one gold paragraph. There were 227 components overall, with a largest size of 21. [Seed-free replay and all depth counts](../../results/nonlinear_retrieval_natural/native_musique_v1/)

HPG job 52970642 built the lexical index, array 52970643 encoded the corpus on four L4 workers, and corrected array 52971603 produced the frozen rankings. The first search attempt used an incompatible positional argument on stable sorting. The correction retained the corpus, embeddings and query fixture. [Corpus builder](../../scripts/prepare_kam_native_musique.py) · [Search harness](../../scripts/run_kam_native_musique.py) · [Audit](../../scripts/analyze_kam_native_musique.py)
