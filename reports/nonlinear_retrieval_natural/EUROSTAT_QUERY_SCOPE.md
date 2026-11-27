# Isolate the requested lookup when two references share a query

Putting a second valid lookup beside the requested one cut full-text dense routing to 220 exact outcomes on 432 address-switch cases, even with the correct date parsed separately. Routing only the marked request recovered all 432. The same controlled change recovered all cutoff-switch cases once the date parser used the marked span. Each pair uses the same words and the same two answers, with the request marker independently crossed with sentence order.

Two types of pairs reuse the official 27,332-record bank and held-out prompts. An **address pair** varies indicator at one country, month and cutoff. A **cutoff pair** varies the publication cutoff at one typed address and keeps only distinct gold releases. There are 108 pairs of each type, with two possible source orders and two possible active roles. The parser sees a line marked “Requested lookup” and a separate background line. A fifth form per pair has two background lines and no declared target. The scope field is explicitly supplied as part of the input grammar.

| Dense document policy | Address exact, of 432 | Address pairs all four | Cutoff exact, of 432 | Cutoff pairs all four |
| --- | ---: | ---: | ---: | ---: |
| Whole text for address, marked span for clocks | 220 | 0/108 | 432 | 108/108 |
| Marked span for address and clocks | 432 | 108/108 | 432 | 108/108 |
| Use first span for address and clocks | 216 | 0/108 | 216 | 0/108 |
| Use last span for address and clocks | 216 | 0/108 | 216 | 0/108 |

Full-text 12-card retrieval reached 229/432 address cases and zero all-four pairs, versus 432/432 and 108/108 with explicit scope. If only the cutoff differs, attaching the correct request metadata fixes the clock branch even with whole-message semantic retrieval. Substituting first-span or last-span clocks in that policy drops to 216 of 432. The exact scope policies abstained on all 216 no-target forms. The literal registry refused the address paraphrases by design. Scope and cutoff masking answer different ambiguities in this fixture. The results condition on a known request field and two indicators present in the bank.

HPG L4 job 53010009. [Reproducible pair construction and runner](../../scripts/benchmark_kam_eurostat_scope.py) · [Numeric paired outcomes](../../results/nonlinear_retrieval_natural/eurostat_scope_v1/) · [Source retrieval study](EUROSTAT_TEXT_ROUTING.md)
