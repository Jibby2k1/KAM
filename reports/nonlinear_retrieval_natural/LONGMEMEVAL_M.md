# Longer conversation histories and marked-turn recovery

On LongMemEval-M, frozen dense retrieval returned every annotated session in the first five on 350 of 470 answerable questions. At a smaller nested history of at most 64 session occurrences, the same method reached 423. Selecting the top 20 turns from the five dense-ranked sessions retained at least one author-marked turn from each required session on 326 questions.

## Source and controls

The official medium release contains 237,655 session occurrences across 500 questions. Exact equality of turn role and text reduced these to 49,018 unique sessions for encoding, while keeping every occurrence's timestamp and annotation mapping. Repeated source session IDs occur in 449 questions. The ranker emits at most one entry per ID and preserves the identity of the actual selected occurrence for turn scoring. The median answerable history had 476 occurrences, 473 distinct source IDs and 10,027 indexed chunks.

The small and medium releases match on all 500 question strings, answers, required session IDs, and required session text. Only three question timestamps are identical. The length control therefore uses nested subsets of the medium release. Each subset keeps all labeled session occurrences and adds background occurrences in one deterministic, question-specific order. This constructs a paired stress test with evidence available at each size. The absolute-date cutoff is a separate interpretation of the supplied timestamps. [Official benchmark](https://github.com/xiaowu0162/LongMemEval) · [Pinned dataset revision](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

| All required session IDs in top five, out of 470 | At most 64 | At most 128 | At most 256 | Full medium history |
| --- | ---: | ---: | ---: | ---: |
| Dense max over all chunks | 423 | 403 | 386 | 350 |
| Dense max over user chunks | 416 | 397 | 378 | 348 |
| Dense max over assistant chunks | 398 | 372 | 341 | 310 |
| Dense with empirical length correction | 415 | 389 | 365 | 326 |
| Dense with a fixed recency bonus | 421 | 400 | 381 | 345 |
| Dense with an absolute cutoff | 389 | 370 | 354 | 324 |

The correction compensates for having more opportunities to produce a large chunk score in longer sessions. It multiplies the empirical tail probability of the session maximum by its chunk count and caps the product at one. It lost 24 successes at full depth (−5.11 percentage points, shared-evidence cluster interval −7.20 to −3.19). A fixed recency bonus had a smaller change and an interval spanning zero. No answerable question's full support fit in the five newest eligible sessions alone.

On the 433 answerable questions with all evidence dated by the cutoff, dense and cutoff-dense each recovered 324 complete sets at full depth. The cutoff excluded at least one required session in 37 answerable questions, including 35 temporal-reasoning items. These cases distinguish author-annotated retrieval from an experimental timestamp policy.

## Evidence inside the selected sessions

| Full medium history | Any marked turn in top 20 | A marked turn for each required session in top 20 |
| --- | ---: | ---: |
| All-turn session ranking | 416 | 326 |
| User-only session ranking | 414 | 330 |
| Length-corrected session ranking | 415 | 307 |

All three rows use the same chunk similarity to select up to 20 distinct turns within five selected sessions. The user-only rule changes session ranking. It recovered 77 complete sets on 121 multi-session items versus 72 for the all-turn rule, while falling from 56 to 52 on the 56 assistant-specific items. These are evidence retrieval and preservation scores, independent of answer generation. The amount of evidence exposed to a reader varies even when all annotated sessions were retrieved.

HPG array 52977284 encoded the eight disjoint session shards on L4 using length-bucketed FP32 E5 batches. Array 52977285 evaluated four independent query shards. [Audited ranks and counts](../../results/nonlinear_retrieval_natural/longmemeval_m_v1/) · [Corpus preparation](../../scripts/prepare_kam_longmemeval_m.py) · [Evaluation](../../scripts/run_kam_longmemeval_m.py)
