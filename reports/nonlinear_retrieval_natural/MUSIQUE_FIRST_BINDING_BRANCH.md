# Early binding alternatives on MuSiQue

A span alternative from the first-ranked paragraph raised complete four-hop candidate coverage from 52 to 57–58 of 160, while final joint exact match stayed at four to five. The five to six net additional full pools produced no additional successes under the frozen gold-paragraph reader. Reading an alternate span from the second-ranked paragraph recovered fewer complete four-hop pools than the primary route.

| Hop count | Questions | Primary full pool / final support / joint | First span from primary paragraph | First span from second paragraph |
| --- | ---: | ---: | ---: | ---: |
| Two | 224 | 183 / 98 / 42 | 185–187 / 96–99 / 39–41 | 181–182 / 88–89 / 37–38 |
| Three | 192 | 120 / 30 / 15 | 126–128 / 33–34 / 17 | 124–127 / 26–28 / 13–14 |
| Four | 160 | 52 / 9 / 5 | 57–58 / 5–9 / 4–5 | 48–50 / 8 / 4 |

The primary reader follows the original author templates and chooses one paragraph per step. Each alternative receives a distinct first answer from a span reader only when its established threshold accepts the span and a later template refers to that binding. It carries its own history, reuses the primary ranking and answer when the rendered queries agree, and fuses active path ranks with the primary path. Each step keeps the original candidate allocation and the final reader receives four distinct paragraphs. Both alternatives preserve the original top 64 and the first expansion tranche. All 1,728 earlier primary traces and final decisions reproduced on the three L4 seed replays.

At four hops the same-paragraph span branch starts on 60–65 questions and adds 143–153 corpus scans. All three paired intervals for the complete-pool gain include zero. Reusing the exact final-reader outcome from the earlier four-gold-paragraph control gives 16 possible joint matches inside the primary pool and 15–16 inside the new pools. No seed added a case the fixed reader could answer under that gold ordering. This finding narrows the headroom for simply expanding first-hop alternatives. The four-paragraph reader and how it combines known evidence are the next controls.

HPG L4 array 53009785. [Verified full and final-evidence outcomes](../../results/nonlinear_retrieval_natural/musique_first_binding_v2/) · [Gold reader control](MUSIQUE_FINAL_READER.md) · [Prior late branch](MUSIQUE_LATE_BINDING_BRANCH.md)
