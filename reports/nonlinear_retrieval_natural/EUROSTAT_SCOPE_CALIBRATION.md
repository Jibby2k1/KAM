# Held-out request wording and revision-scope abstention

Ranking two already segmented clauses with frozen E5 selected the correct Eurostat request 648 of 864 times. Masking the date-bearing statistical request before scoring its surrounding language raised forced ranking to 864. A relative confidence threshold also handled every ambiguous case when the two clauses shared identical role wording. With different wording, the same approach traded exact answers against refusal. The quality of parsing request roles limits otherwise exact as-of lookup.

The fixtures use official Eurostat releases and authored clause-role language. Fit covers four wording families, calibration covers two separate families, and test covers six other families and held-out countries. Each of 216 test bundles swaps the positions and active role of two valid address or cutoff lookups. This gives 864 single-request examples. An additional 216 have two background clauses and 216 have two active clauses. The matched ambiguity panel uses the same role phrase in both clauses. The mixed panel selects a different template family for the second clause of each ambiguous case. All hashes, lookup targets, and single-request text are held fixed between the two ambiguity panels.

The cue-only transformation identifies the sentence with both an explicit observation month and cutoff and replaces it with one constant sentence. It leaves the contextual request wording and clause positions intact. The method assumes the two clauses have already been segmented. When a policy chooses a clause, the original clause is still used for address and exact as-of lookup. Correct scope yielded the exact snapshot throughout. Absolute calibration labels each clause using a shared threshold. Relative calibration selects the higher-scoring clause when the gap exceeds a threshold. Candidate thresholds maximize mean exactness of single, neither and both classes on calibration only. FLAN-T5-base chooses the highest-scoring A, B or abstain token without fitting.

| Ambiguity panel and input | Policy | Exact single / 864 | Correct decline, neither / 216 | Correct decline, both / 216 |
| --- | --- | ---: | ---: | ---: |
| Matched, full clauses | Frozen prototype, forced | 648 | 0 | 0 |
| Matched, full clauses | Frozen prototype, absolute | 4 | 216 | 214 |
| Matched, full clauses | Trained E5 linear score, absolute | 146 | 194 | 160 |
| Matched, full clauses | Frozen prototype, relative | 176 | 212 | 108 |
| Matched, full clauses | FLAN-T5-base choice | 411 | 0 | 0 |
| Matched, cue only | Frozen prototype, absolute | 576 | 216 | 216 |
| Matched, cue only | Frozen prototype, relative | 864 | 216 | 216 |
| Mixed, full clauses | Frozen prototype, relative | 144 | 168 | 140 |
| Mixed, cue only | Frozen prototype, relative | 432 | 216 | 216 |
| Mixed, cue only | Trained E5 linear score, relative | 720 | 216 | 86 |
| Mixed, cue only | Trained E5 linear score, absolute | 288 | 216 | 102 |
| Mixed, cue only | Frozen prototype, forced | 864 | 0 | 0 |
| Either, oracle role | Exact reference | 864 | 216 | 216 |

The matched cue-only comparison deduplicates to six request and six background phrases. Identical same-role claims have equal embeddings, letting a zero-gap threshold separate mirrored ambiguities. On mixed cues, the relative frozen policy retains perfect ambiguity rejection at 50% single-request coverage. The relative trained policy answers 720 correctly and defers 144, while accepting 130 requests that contain two active clauses. Results characterize transfer between these authored families. Unstructured messages, indirect requests, and clause boundary detection require independent evaluation. Report exact release along with refusal behavior whenever the input can contain multiple valid instructions.

HPG L4 arrays 53013055 and 53013058. An earlier frozen replay matched all 2,592 case outcomes for each existing policy. [Paired decisions, calibration and per-family counts](../../results/nonlinear_retrieval_natural/eurostat_learned_scope_v3/) · [Same-word marked scope](EUROSTAT_QUERY_SCOPE.md) · [Typed as-of lookup from text](EUROSTAT_TEXT_ROUTING.md)
