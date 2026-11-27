# Recursive MuSiQue search and the intermediate-answer oracle

On a fresh 160-question four-hop MuSiQue slice, recursive paragraph conditioning recovered every support on five questions. Three parallel anchors recovered nine and hybrid lexical and dense search recovered 12. Author-provided subquestions recovered every support on 13 questions with references removed and on 95 questions with author-provided intermediate answers inserted. This isolates substantial headroom in identifying the entities that later searches depend on.

## Fixed candidate budget

All methods used the same 101,748-paragraph bank and sealed 576-question [MuSiQue fixture](NATIVE_MUSIQUE.md). Both new article-conditioned methods began with the archived dense top 64 and identical first 22 conditioned additions. They then filled two slots of 21 new paragraphs each after excluding all seen IDs. The parallel method conditioned on the second and third initial dense hits. The recursive method added the first newly admitted result to each successive query and included bounded prefixes of up to three previously chosen documents. Both methods use the same E5 encoder and three conditioned branches, with dependent search stages for the recursive method. All use 128 final document slots. We did not time matched single requests.

The annotated decomposition controls also retained the original dense 64. Each split of 64 additional slots was allocated equally across the two, three, or four author subquestions, with round-off assigned to the earliest hop. One control deleted numbered references and the other inserted the author-provided answers from the referenced earlier steps. These are evaluation oracles using author decomposition and, in the latter arm, gold intermediate answers. They measure paragraph recovery under supplied subqueries.

| Complete support chain at depth 128 | 2 hops, 224 | 3 hops, 192 | 4 hops, 160 | All, 576 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 130 | 39 | 7 | 176 |
| Hybrid lexical and dense | 135 | 56 | 12 | 203 |
| One conditioned anchor | 144 | 58 | 6 | 208 |
| Two conditioned anchors | 146 | 59 | 7 | 212 |
| Three parallel anchors | 148 | 55 | 9 | 212 |
| Three recursive anchors | 148 | 58 | 5 | 211 |
| Author subquestions, references removed | 140 | 72 | 13 | 225 |
| Author subquestions, gold answers substituted | 204 | 164 | 95 | 463 |

The recursive method changed complete recovery by −0.17 percentage points overall versus three parallel anchors, with a shared-evidence cluster bootstrap interval of −2.17 to +1.97. On four-hop questions the paired change was −2.50 points (−6.07 to +0.61). Hybrid recovered 332 of 640 individual four-hop support paragraphs, compared with 267 for the three parallel anchors and 258 for recursive. The gold substitutions raised four-hop complete recovery by 51.25 points versus the same unresolved templates (+33.75 to +66.46). The four-hop annotation target appeared in the corresponding top 16 on 53, 47, and 53 questions at hops two through four with the unresolved templates. Those counts became 140, 114, and 142 with resolved references.

The executed search was HPG array 52974274 across four L4 workers. The evaluation verified the frozen input hash, all 576 matching question IDs, common first 64 ranks in each conditioned method, 128 distinct candidates, and all annotated reference orderings against the source. This study measured paragraph retrieval. [Candidate ranks and paired results](../../results/nonlinear_retrieval_natural/musique_recursive_v1/) · [Search runner](../../scripts/run_kam_musique_recursive.py) · [Paired analysis](../../scripts/analyze_kam_musique_recursive.py)
