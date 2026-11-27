# Route the address, then select the as-of release

On 864 held-out, answerable Eurostat requests, frozen E5 retrieval returned the exact snapshot at rank one 114 times. Taking only the typed country and indicator from that same document, then applying the parsed reference month and an exact publication-cutoff lookup, recovered all 864 snapshots. Filtering past and same-month records before dense ranking reached 339 exact, although it selected the correct address on all 864. Under this prompt grammar, ranking was a useful address detector and a poor rule for choosing among valid revisions at that address.

## Source and evaluation

The full bank contains 27,332 distinct official publication cells for two indicators across the **same** six countries. Both indicators are available for each ambiguous request. Retrieval sees ordinary statements that carry the country, indicator, reference month, publication date, and numerical payload. SQLite verifies the labels directly from both source tables. The independent evaluation script reconstructs the same as-of selection by binary search.

Thresholds use 2023 requests for Hungary, Ireland and Portugal. Held-out prompts use 2024 through June 2025 for Romania, Slovenia and Slovakia and different paraphrase families. Answerable prompts use literal terms, English paraphrases, two-letter country codes, or explicit 90/365-day offsets. The visible date grammar compiles one month and either an ISO cutoff or a fixed day offset after month end. The benchmark covers authored wording over official data. It does not estimate performance on arbitrary prose or indicators outside this archive. [Eurostat unemployment vintages](https://ec.europa.eu/eurostat/databrowser/view/ei_lm_m_vtg/default/table?lang=en) · [Eurostat industrial vintages](https://ec.europa.eu/eurostat/databrowser/view/ei_is_m_vtg/default/table?lang=en)

| Policy | Exact snapshots, of 864 | Correct address with month | Returns within cutoff |
| --- | ---: | ---: | ---: |
| BM25 document at rank one | 95 | 384 | 763 |
| BM25 document after month and cutoff filter | 328 | 679 | 864 |
| Filtered BM25 address followed by exact lookup | 679 | 679 | 864 |
| Dense document at rank one | 114 | 724 | 421 |
| Dense document after month and cutoff filter | 339 | 864 | 864 |
| Unfiltered dense address followed by exact lookup | 864 | 864 | 864 |
| Dense country-and-indicator card followed by exact lookup | 838 | 838 | 864 |
| Literal country and semantic indicator followed by exact lookup | 864 | 864 | 864 |

An unfiltered embedding often recognized the subject while selecting a future or wrong-month record. Applying both constraints made the correct record appear in all top-64 E5 and BM25 lists. A filter still leaves multiple eligible releases at one address. An exact latest-publication operator performs that final choice. Dense first-document address plus exact lookup gained +86.81 points over direct reading, with a paired [83.33, 89.93] interval from resampling country, reference month and indicator groups. These rows measure exact snapshot selection and expose numeric agreement separately in the replay.

## Abstention

The text route first recognizes the country from a code or canonical name. Literal indicator phrases resolve directly. Other indicator phrases compare two frozen semantic card embeddings. The held-out prompts separately test unsupported countries, unsupported indicators with recognized countries, and missing indicator names when both measures are valid. Separate thresholds on top cosine and margin fit these last two cases on different development negatives.

| Policy | Exact answerable | Ambiguous abstentions | Unsupported indicator abstentions | Unsupported country abstentions |
| --- | ---: | ---: | ---: | ---: |
| Literal registry | 432/864 | 108/108 | 108/108 | 108/108 |
| Semantic indicator, always choose after country match | 864/864 | 0/108 | 0/108 | 108/108 |
| Top-score unknown gate | 648/864 | 0/108 | 108/108 | 108/108 |
| Margin collision gate | 864/864 | 108/108 | 0/108 | 108/108 |
| Both gates | 648/864 | 108/108 | 108/108 | 108/108 |

The unknown gate declined half of the paraphrase and country-code requests. The positive dates, wordings and unsupported examples are repeated within each family. The hundreds of decisions show the effect of these controls over different real releases and countries, with a much smaller number of distinct unsupported indicator phrases. Calibrating these gates again after changing the set of possible indicators is an open experiment.

HPG L4 arrays 53008596 and 53008597. [Pinned numeric source responses, fixture and result replays](../../results/nonlinear_retrieval_natural/eurostat_text_routing_v1/) · [Exact clocks and learned payload control](ADDRESS_LOCAL_CLOCK.md)
