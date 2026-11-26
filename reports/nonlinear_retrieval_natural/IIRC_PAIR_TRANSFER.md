# Frozen pair scoring transfers unevenly to IIRC

On 153 IIRC development questions requiring two linked articles, frozen conditioned sequential retrieval recovered **40** gold pairs. Independent top-two recovered **32**, the frozen one-pass pair scorer **22**, and the pair scorer with between-candidate title features masked **30**. All policies saw the same original paragraph and the same visible hyperlink pool. [IIRC paper](https://aclanthology.org/2020.emnlp-main.86/) · [Audited results](../../results/nonlinear_retrieval_natural/iirc_pair_v1/summary.json)

| Annotated evidence group | Questions | Independent top-two | Sequential | Pair scorer | Pair scorer, title features masked |
| --- | ---: | ---: | ---: | ---: | ---: |
| All two-link questions | 153 | 32 | **40** | 22 | 30 |
| Between-gold title mention | 33 | 7 | 13 | **16** | 7 |
| No between-gold title mention | 120 | 25 | **27** | 6 | 23 |

Sequential gained **5.23 percentage points** over independent top-two (paired article-bootstrap 95% interval **+1.36 to +9.43**). The pair scorer fell **11.76 points** below sequential (**−18.67 to −5.56**). The effect of enabling title mentions varied across strata. It gained 9 net successes on the 33 questions where one gold article mentioned the other title in its visible prefix (**+27.27 points**, **+10.00 to +45.16**) and lost 17 on the other 120 (**−14.17 points**, **−21.24 to −8.06**). These strata use gold annotations to diagnose the policy. An oracle choosing between the pair and sequential policies would recover **44** pairs, four more than sequential alone. Intervals resample whole source articles and retain all their questions. [Per-question replay and counts](../../results/nonlinear_retrieval_natural/iirc_pair_v1/)

## Cohort and controls

The official development release has 1,301 questions from 430 source articles. Of 954 answerable questions, 153 require exactly two annotated linked articles. They come from 117 source articles and have 8–23 visible candidate links per question, with a median of 13. Both annotated linked titles were among the resolved candidates on all 153 questions. We use the answerer's context annotations for gold links. The question writer's proposed links agreed exactly with those titles on 832 of all 954 answerable development questions. The other answerable questions and the 347 labeled unanswerable are outside this pair-recovery experiment. There is no train/development overlap in source article titles or question IDs within IIRC, and 193 annotated linked titles occur in both splits. [Source audit](../../results/nonlinear_retrieval_natural/iirc_pair_v1/source-audit.json)

Each query contains the question and up to 100 words of the source paragraph. Candidate keys use the linked article title and a lead excerpt, with the same frozen E5 encoder and HotpotQA-trained query adapter in all arms. Sequential uses the top independent hit, appends the first 48 words of that hit's title and text to its query, and averages the two score vectors before choosing a distinct second article. The pair scorer uses the frozen HotpotQA head over every unordered pair. Its link features are exact between-candidate title mentions in the first 120 word tokens. The masked arm zeroes those inputs while retaining the head, and all arms still search the same official source hyperlinks. The metric tests annotated article recovery. This run supplies no answer-generation or supporting-span result. [Fixture preparation](../../scripts/prepare_kam_iirc_pair.py) · [Inference driver](../../scripts/run_kam_iirc_pair.py)

HPG array **52954845** used four B200 shards. Median warm route times were 4.25 ms independent, 8.70 ms sequential, 4.64 ms pair, and 4.56 ms masked. The clock includes query encoding and routing after the candidate key and link graph are prepared. It excludes article reads, offline candidate encoding, model initialization, and answer decoding. The timing audit randomizes policy order by question.

## Reproduce

The archived numeric replay retains ordered candidate-pool fingerprints, candidate slots, and timing. To regenerate inputs from the [authors' download links](https://raw.githubusercontent.com/jferguson144/IIRC-baseline/main/setup.sh), use the official development JSON and linked-article archive with `ijson` installed.

```bash
study=results/nonlinear_retrieval_natural/iirc_pair_v1
python scripts/prepare_kam_iirc_pair.py --dev iirc/dev.json --articles context_articles.tar.gz \
  --output /tmp/kam-iirc-fixture.json --manifest /tmp/kam-iirc-manifest.json
diff -u "$study/manifest.json" /tmp/kam-iirc-manifest.json
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_iirc_pair.py "$study/replay.json.gz" \
  --manifest "$study/manifest.json" --source-audit "$study/source-audit.json" \
  --output /tmp/kam-iirc-pair.json
diff -u "$study/summary.json" /tmp/kam-iirc-pair.json
```
