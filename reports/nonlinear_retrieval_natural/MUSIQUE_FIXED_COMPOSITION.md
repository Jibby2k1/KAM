# Does an explicitly bound question improve four-passage reading?

Replacing the original MuSiQue question with its final author step reduced exact accuracy under a matched four-paragraph prompt. On 160 cases with all gold paragraphs supplied, the unchanged question got 48 exact answers. Filling the final step with predicted bindings got 29 and using gold intermediate values got 35. The same comparisons made little difference on passages selected by the primary route, where only nine questions had all four gold paragraphs.

| Fixed evidence, 160 cases | Original question | Final step, unbound | Final step, predicted bindings | Final step, rotated binding control | Final step, gold bindings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four gold paragraphs | 48 | 23 | 29 | 23 | 35 |
| Primary four selected paragraphs | 20 | 13 | 16 | 15 | 19 |

The fixed FLAN-T5-base model receives the same four passages in the same order for every question variant. Each passage is capped at 72 tokens. Masked slots reserve the same question prefix positions, keeping the actual passage tokens and context positions matched across variants. Earlier 96-token original-question outcomes replayed exactly at 59 on gold evidence and 22 on primary evidence. That baseline uses the archived serialization and a different passage cap.

The predicted bindings came from reading each prior author subquestion with its assigned passage. The gold condition uses annotation-based role assignment and the separate gold prefix bindings. The rotation moves the same predicted answers to different reference indices. The unit of evaluation remains the answer to the original question. The final authored answer agrees with an accepted original answer for every case. Two last subquestions directly reference an intermediate label equal to the final answer. Excluding those two leaves the ranking of the five results unchanged. [Verified passage and answer replay](../../results/nonlinear_retrieval_natural/musique_reader_composition_v1/)

For gold passages, the paired change for predicted bindings versus the matched original question was −11.88 percentage points [−29.63, +2.48] by shared-evidence clusters. Gold bindings versus original changed −8.13 [−24.10, +5.81]. The unbound and rotated controls showed larger evidence-backed losses. On primary-selected passages, the original question got one joint match under the matched prompt and the other four variants got zero. The results support keeping the original question as the readout reference and testing passage coverage and reader capacity independently.

HPG L4 array 53011105. [Early binding and gold-readout headroom](MUSIQUE_FIRST_BINDING_BRANCH.md)
