# Temporal retrieval on natural prose and archived revisions

The studies cover three grades of temporal supervision. TimeQA supplies prose and annotated spans. TempReason supplies explicit fact intervals. The Philadelphia Fed archive supplies observation and publication values with independent coordinates. Keeping those settings distinct identifies when a model needs to interpret language and when exact lookup can enforce the required semantics.

## TimeQA prose and same-entity date collisions

Froze 296 development entities and 320 test entities from the authors’ crowd-verified annotations. Each entity contributed two questions for disjoint periods, with different answers in different supporting paragraphs. A small ranker used the dated and undated E5 affinities, entity and token overlap, plus four features measuring the occurrence and proximity of years. It was compared with dense retrieval and matched training without date features.

| Clean evaluation on 640 questions | Support selected | Answer EM | Joint EM | Both periods’ support for an entity |
| --- | ---: | ---: | ---: | ---: |
| Dense without date | 43 | 33 | 11 | 0 |
| Dense with date | 117 | 80 | 49 | 13 |
| Learned ranker without date | 62 | 43 | 19 | 0 |
| Learned date-aware ranker | 261 | 184 | 126 | 61 |

The date-aware ranker improved evidence by 22.5 points over dense, with a paired entity-bootstrap interval of +18.3 to +26.7. The gold-passage reader answered 316/640. Adding eight other-entity and same-relation passages changed the date-aware result by only one question. That perturbation had little effect for all four selectors.

A harder intervention used the same entity’s wrong-period passage, prefixed with an archival review period whose date describes document review. In the sham control its date was unrelated. In the collision control it matched the question’s date. The date-aware ranker selected both variants from fixed pools with the same eight added slots. On 584 questions from 292 eligible entities, its support selections fell from **233 to 183** and distractor selections rose from **12 to 132**. Dense evidence fell from 90 to 81 and distractor selections rose from 17 to 58. A year in a document can be easy to detect and irrelevant to the event time. [Period contrast results](../../results/nonlinear_retrieval_natural/timeqa_contrasts_v1/summary.json) · [Matched date-collision results](../../results/nonlinear_retrieval_natural/timeqa_collisions_v1/summary.json) · [Author data](https://github.com/wenhuchen/Time-Sensitive-QA)

On the authors’ original test questions, 827 of 989 had a usable annotated answer and visible gold span. Easy and hard forms share the same evidence. The learned date-aware ranker remained frozen, with zero fitting-entity overlap. It recovered support on 357 easy and 235 hard questions, compared with 183 and 159 for dense. Corresponding joint exact matches were 146 and 86 for the learned ranker, and 70 and 55 for dense. The constructed and authored evaluation cohorts share 38 held-out entities. Removing those 154 questions leaves 673 authored questions across 212 entities. The hard-form support counts in this independent slice were 186 versus 135, a gain of +7.6 points with a paired entity-bootstrap interval of +3.1 to +12.0. [Human-question transfer](../../results/nonlinear_retrieval_natural/timeqa_human_v1/summary.json) · [Entity-disjoint slice](../../results/nonlinear_retrieval_natural/timeqa_human_v1/independent-entities-summary.json)

An attempted fix fit matched heads using dates from the single sentence most semantically similar to the query, or affinity-weighted dates across all sentences. E5 compared sentences and questions with years removed. The original all-years control refit bit-for-bit. From sham to collision, support fell from 173 to 163 with the best-sentence rule, and from 109 to 94 with the weighted rule. The smaller drop came with worse absolute support than the all-years head (233 to 183). Clean support from the best-sentence rule was 234/640 versus 261/640 for all years. A sentence’s similarity to the question gives an imperfect proxy for whether its date describes the event. [Date-role diagnostics](../../results/nonlinear_retrieval_natural/timeqa_date_roles_v1/summary.json)

## TempReason fact text

Sampled one question per entity within each official test level, with 384 absolute-date and 384 relative-event questions. The no-training rule parses dates from input facts and the visible question. The relative parser locates the named event and selects the neighboring start date in the requested direction. The scoped version restricts the same rule to the entity named in the question. Both versions use the same two-passage reader budget. Eight facts drawn from other entities with the same relation were appended as noise.

| Task and policy | Clean evidence / answer EM | Noisy evidence / answer EM |
| --- | ---: | ---: |
| Absolute date, dense top two | 320 / 147 | 320 / 147 |
| Absolute date, explicit time | 384 / 358 | 384 / 313 |
| Absolute date, time and scope | 384 / 358 | 384 / 358 |
| Relative event, dense top two | 133 / 110 | 133 / 110 |
| Relative event, explicit time | 380 / 327 | 252 / 219 |
| Relative event, time and scope | 380 / 327 | 380 / 327 |

Evidence means that at least one answer string was contained in a selected fact. Answer EM accepted any author-supplied answer. The rules exploit the standardized fact syntax and ten known relation templates. Their performance is an interpretable control on this input format. Gold-fact answer ceilings were 359/384 and 334/384. [Absolute and relative temporal benchmark results](../../results/nonlinear_retrieval_natural/tempreason_v1/summary.json) · [Author data](https://github.com/DAMO-NLP-SG/TempReason)

## Executable labels for fictional event schedules

UnSeenTimeQA exposes templated logistics event histories and set-valued answers. An independently implemented parser and state replay matched the **entire answer set on all 3,600** released easy and medium questions with static or relative time. The solver uses the author’s disclosed conventions for cargo membership and for what locations count during loading and unloading. It handles midnight rollovers and recovers the associated event indices. Independently shifting every timestamp in a question by a random minute offset preserved both the denotation and recovered event on all 3,600. This verifies an inexpensive noise axis with exact labels on serial questions with explicit event clocks. [Label audit](../../results/nonlinear_retrieval_natural/unseentimeqa_v1/summary.json) · [Author paper](https://aclanthology.org/2025.acl-long.94/)

The event-retrieval probe used 600 questions spanning ten distinct initial worlds and one timed target event per question. The added event comes from a different world, mentions the same package symbol, and keeps its action and duration. Only its clock shifts between the compatible and sham variants. A time-only guard found the target on 600/600 sham inputs and 123/600 collision inputs. It returned both candidates in the top two across all 600 collisions. Adding the known world ID restored 600/600 top-one. Frozen dense retrieval recovered only 50/600 clean targets, rising to 60 after directly resolving relative times in the query. This probe uses exact event indices and tests retrieval eligibility. Answering the underlying question also requires replaying package state through earlier events. [Matched event and provenance probe](../../results/nonlinear_retrieval_natural/unseentimeqa_probe_v1/summary.json)

The duration-only hard serial audit reconstructed the schedule from a single named event's clock and the ordered event durations. It matched the complete published answer set on all 900 static and 900 relative questions. For 900 hypothetical questions, it matched 893 and abstained on seven whose changed event description matched multiple records. On those 893 resolved cases, ignoring the duration intervention gave 408 exact matches and 140 abstentions. A common clock shift preserved the answer and inferred event indices on every resolved case, and the target's index relative to the named event agreed with author depth throughout. Each question type spans ten distinct initial worlds and ten distinct action sequences. This provides an executable test for duration changes and clock translation within those serial schedules. [Replay script](../../scripts/audit_kam_unseentimeqa_serial.py) · [Source revision and file hashes](../../results/nonlinear_retrieval_natural/unseentimeqa_hard_serial_v1/summary.json) · [Author dataset revision](https://github.com/nurakib/UnSeenTimeQA/tree/84f910fcd32f2b21d6e34d9efb98e94eb04844bb/data/hard_serial)

### Parallel schedules

The concurrent audit inferred precedence from vehicle and package identities in the narrated actions. It allowed matching load or unload actions on the same vehicle to start together, then assigned each group its first eligible start. When a query matched exactly one distinct set of locations across active package events, that set was returned. A separate diagnostic reconstructed the authors' ten published plan schedules by reading the start-time formulas and applying them to the durations in each question. This diagnostic used the published dependency schedules as extra information. [Audit script](../../scripts/audit_kam_unseentimeqa_parallel.py) · [Source fingerprints and per-plan counts](../../results/nonlinear_retrieval_natural/unseentimeqa_hard_parallel_v1/summary.json) · [Author source](https://github.com/nurakib/UnSeenTimeQA/tree/84f910fcd32f2b21d6e34d9efb98e94eb04844bb/data_generation)

| Schedule and answer rule | Static, 900 | Relative, 900 | Hypothetical, 900 |
| --- | ---: | ---: | ---: |
| Serial schedule control | 70 | 75 | 75 |
| Inferred concurrent schedule, unique active-event answer | 870 | 885 | 863 |
| Published plan schedule, unique active-event answer | 894 | 891 | 890 |
| Published plan schedule with annotated target event | 900 | 900 | 890 |

The inferred and published schedules agreed on every timestamp for 2,274 of the 2,690 cases whose event references were unique. Using the published schedules, 15 cases had simultaneous eligible events with different sets of locations. The release's depth metadata identified the target event and reproduced all 2,690 resolved labels, including those 15. The other ten hypothetical questions named a changed event that occurred more than once. Clock translation preserved the inferred results on every resolved case. The ten reused plans limit independent variation, so use complete-plan holdout and audit label uniqueness when generating further questions.

### Duration changes with the same numbers

Constructed 1,173 pairs from the hard serial static and relative questions across ten plans. Each intervention swapped two duration values on opposite sides of the original target, changing the selected event and location set. The paired sham swapped equally different duration values strictly after both targets. Question wording, event order, the multiset of durations, and the number of edited events stayed fixed. The median duration difference was 35 minutes. An independent cumulative schedule plus the original package replay agreed on the new targets. The reported slice contains eligible cases where both controlled edits could be constructed.

| Frozen E5 selector | Original top one | Sham top one | Intervention top one | Both original and intervention top one |
| --- | ---: | ---: | ---: | ---: |
| Individual events and full question | 1 | 1 | 22 | 0 |
| Individual events and target clause | 5 | 6 | 52 | 0 |
| Five-event windows and full question | 15 | 14 | 19 | 0 |
| Five-event windows and target clause | 85 | 82 | 38 | 0 |
| Individual events with durations masked | 1 | 1 | 16 | 0 |

The best clean selector's top-five counts were 330, 332, and 275 across the three forms. That selector changed its top-one choice on 44 interventions and 22 shams. The independent target came from the sequence's cumulative clock, so local semantic windows cover only part of the required computation. These counts evaluate exact target event retrieval in the edited generated plans. They provide a control for a learned ordered selector and say nothing about natural-text answer decoding. HPG L4 array 52963861 supplied the scores. [Fixture builder](../../scripts/prepare_kam_tempreplay_swaps.py) · [GPU baseline](../../scripts/run_kam_tempreplay_swaps.py) · [Audit and plan-level counts](../../results/nonlinear_retrieval_natural/unseentimeqa_duration_swaps_v1/summary.json)

## Real revision and abstention

Read four linked observation-by-vintage tables from the Philadelphia Fed archive, totaling **190,764** nonmissing cells. For each question the exact target was the latest cell from the requested series and observation that was available by the vintage cutoff. A separately implemented SQLite lookup agreed on every fixture label. Unavailable questions have no corresponding cell by the cutoff. The numerical target is the archived representation for that edition. Rebasing can change earlier observations.

The 348 training groups used observations from 1978–2006 and withheld nominal consumption as a fitting class. The 160 test groups used observations from 2013–2022 and cover all four series. Each test group has earlier, later, and unavailable cutoffs rendered in a canonical and a held-out paraphrase. The prepared candidate test added 32 later revisions and neighboring records to a shared 16-record pool.

| Query surface and status | Dense exact | Learned typed ranker exact | Semantic identity plus exact date guard |
| --- | ---: | ---: | ---: |
| Canonical known, 320 | 208 | 320 | 320 |
| Canonical unavailable, 160 | 0 | 160 | 160 |
| Paraphrased known, 320 | 130 | 80 | 80 |
| Paraphrased unavailable, 160 | 0 | 160 | 160 |

The date guard eliminated future selections under both surfaces. The canonical alias mapping was 320/320 on known queries, but the held-out descriptive names resolved correctly for only 80/320. The learned ranker abstained on 164 answerable paraphrases after identity resolution failed. The remaining limitation is entity identity, separately from enforcing time and arrival constraints. A dense similarity threshold calibrated on earlier canonical questions was particularly brittle on the held-out paraphrases. [Exact revision and abstention results](../../results/nonlinear_retrieval_natural/vintage_revisions_v1/summary.json) · [Philadelphia Fed complete vintage archive](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-full-time-series-history)

A further 128 held-out compositional series names tested whether splitting the name into measure and dollar basis improved identity resolution. E5 resolved 64/128 and the factored scorer resolved 65/128. A separately calibrated unknown gate rejected 23/24 unsupported identities and left 50/128 known queries correct. A further collision gate rejected 32/32 ambiguous names and left 11/128 known queries correct. This quantifies the coverage cost of the two gates. It also shows that stronger date rules would have little effect on this identity failure. [Alias and abstention audit](../../results/nonlinear_retrieval_natural/vintage_aliases_v1/summary.json)

## Corpus-wide revision retrieval and hardware

The same questions were also tested against all 190,764 archive cells using a 293 MB FP32 E5 matrix and exact scores. Filtering only to the requested observation raised canonical top-64 inclusion from 304/320 to 320/320 and paraphrased inclusion from 174/320 to 320/320. Applying both observation and publication constraints raised canonical top-one selection from 76 to 230. The separate typed index still depended on a correct identity, giving 320 canonical and 80 paraphrased exact matches. [Full-archive retrieval](../../results/nonlinear_retrieval_natural/vintage_global_v1/summary.json)

A guarded top-k with the same within-device tie rule matched full sorting on all 960 queries and all four scoring arms per GPU. The L4 and Blackwell top-one outputs agreed across all queries. Complete top-64 ordering for raw dense scores agreed for 887 of 960, compared with 957 under both guards. At batch 64, L4 dense median scoring and selection fell from 9.973 to 2.381 ms and the Blackwell result fell from 1.753 to 0.537 ms. The faster implementation was slower at batch one. These are resident-corpus scores and selection, excluding query encoding and disk access.

## Separate local ingestion time

Replayed the 190,764 archived cells while sampling a separate local arrival clock. The publication timestamps and values come from the workbook. Arrival times are simulated delays. Four seeds each generated 200,000 point-clock queries and 200,000 independently sampled clock pairs. An independently validated descending scan, a minimum-arrival tree, a SQLite index, and a live index agreed exactly on all 1.6 million queries. A vintage-only index disagreed on 241,832 of 800,000 point-clock queries and 311,634 of 800,000 independent clock pairs. Median tree lookup was 0.723 and 0.984 microseconds across the two panels, compared with 3.127 and 3.199 microseconds for the Python scan. The live index took about 0.44–0.45 microseconds per insert and 0.55–0.57 per query in this process-resident CPU simulation. These timings describe the benchmark implementation on HPG CPUs. [Stream replay and parity](../../results/nonlinear_retrieval_natural/vintage_stream_v1/summary.json)

## Conversational memory and annotation clocks

Audited all 500 published LongMemEval-S questions. A literal comparison of the released question and session timestamps found 76 questions with at least one session dated after the question and 44 for which a cutoff would hide some annotated evidence. Forty-one of those 44 questions were in the study’s 470 answerable items. The cutoff is an experimental assumption about the meaning of the timestamps. Their interpretation and annotation policy should be verified before using this file as an exact as-of retrieval test.

Using the published evidence regardless of date, maximum E5 chunk score recovered every annotated session in the top five on 430/470 answerable questions. BM25 recovered 390/470. Among the 429 answerable questions with complete evidence before the cutoff, dense retrieved all evidence in 392 cases. A synthetic conversation with strong lexical overlap reduced that to 376 and entered the top five on every case. Dating the added session after the cutoff restored exactly the original guarded ranks in all 429 cases. Dating the same text before the cutoff left the score at 376. These counts measure exact annotated session selection under a controlled lexical echo. [LongMemEval-S probe](../../results/nonlinear_retrieval_natural/longmemeval_v1/summary.json) · [Author benchmark](https://github.com/xiaowu0162/LongMemEval)

## Seeded lifelogs and conflicting provenance

The TimelineQA author generator produced 1,025,739 artificial episodes across 24 independent personas and three density settings. From each persona, the probe retained six date-specific atomic questions for each of four categories. Each of the 576 cases had one annotated event and at least one naturally occurring wrong-day episode with the same question template and a different answer. The smallest supplied pool includes all events recorded that day. Larger 64- and 256-episode pools add seeded background episodes. Compare a true journal-date prefix with an otherwise identical prefix assigning the query date to one wrong-day episode. The underlying narrator, event content, authoritative day and gold event ID are unchanged.

| Top-one support by policy | True dates, 16 | False date, 16 | True dates, 256 | False date, 256 |
| --- | ---: | ---: | ---: | ---: |
| Dense text retrieval | 504 | 329 | 438 | 312 |
| Dense restricted to text-reported day | 522 | 330 | 522 | 330 |
| Dense restricted to authoritative day | 522 | 522 | 522 | 522 |

At 256 candidates, the false prefix cost the text-reported-date policy 33.3 percentage points of support (95% persona-bootstrap interval −36.6 to −30.0). The authoritative guard matched every original selection with no wrong-day top-one entries. It still missed 54 gold episodes because several events can share the day. These are exact event retrieval counts for a derived atomic probe with generated metadata and author-provided evidence IDs. [TimelineQA probe](../../results/nonlinear_retrieval_natural/timelineqa_v1/summary.json) · [Author benchmark](https://aclanthology.org/2023.findings-acl.6/)
