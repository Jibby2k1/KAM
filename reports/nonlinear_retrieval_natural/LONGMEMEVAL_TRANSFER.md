# Wikipedia decomposition transferred to conversation memory

A decomposer trained on MuSiQue Wikipedia questions reduced complete top-five session recovery on LongMemEval-M. Across three seeds, dense retrieval recovered every required session on 350 of 470 answerable questions. The bound decomposition recovered 334, 338 and 342. Temporal-reasoning questions accounted for much of the loss.

The experiment used three frozen MuSiQue checkpoints. A valid plan of two to four backward-referencing steps kept one original dense starter and filled four additional slots in step order with new session IDs. The unresolved control stripped placeholders from each search query. The bound arm estimated the missing value from the most relevant conversation chunk with a fixed FLAN-T5-base reader. The same conversation bank, encoder, session budget, and top-20 turn ranker were used throughout. Invalid plans returned the original dense ranking. No LongMemEval answers, timestamps or gold sessions entered the plan, binding or selection. HPG L4 arrays 52980423 and 52980424 generated the plans and evaluated the policies.

| Policy | All required sessions, out of 470 | Marked turn for each session, out of 470 | Temporal complete sessions, out of 127 |
| --- | ---: | ---: | ---: |
| Dense | 350 | 326 | 82 |
| Unresolved, seed 106331 | 333 | 312 | 71 |
| Bound, seed 106331 | 334 | 312 | 71 |
| Unresolved, seed 106339 | 335 | 318 | 72 |
| Bound, seed 106339 | 338 | 319 | 74 |
| Unresolved, seed 106367 | 341 | 318 | 77 |
| Bound, seed 106367 | 342 | 318 | 76 |

Valid plans were returned on 161, 182 and 167 answerable questions respectively. The bound arm's overall complete-session change was −3.40, −2.55 and −1.70 percentage points. Shared-evidence cluster bootstrap intervals were [−5.33, −1.70], [−4.47, −0.65] and [−3.42, 0.00]. Temporal complete-session changes were −8.66, −6.30 and −4.72 percentage points, with all three intervals below zero. Replacing dense slots with generated-query hits caused more losses than the reader recovered through binding. This comparison measures evidence retrieval and preservation. Answer generation from the selected turns is unmeasured.

The next comparison should preserve the dense ranking as a fallback, check fixed candidate additions before reranking back to five, and distinguish author-reported event dates from conversation timestamps.

[Medium-history baseline](LONGMEMEVAL_M.md) · [Numerical replay](../../results/nonlinear_retrieval_natural/longmemeval_m_transfer_v1/) · [Evaluator](../../scripts/run_kam_longmemeval_m_transfer.py)
