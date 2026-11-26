# Frozen reader with fixed IIRC article identities

The frozen FLAN-T5-small reader answered **25 of 134** IIRC questions exactly with the original source lead and the leads of both gold linked articles. It reached **26** when the linked windows were chosen using the annotations, and **38** with only the annotated spans from all three inputs. The annotated-span gain over the three leads was **+9.70 percentage points** (paired 95% source-article interval **+1.52 to +18.11**). All reader arms used the same short-answer instruction and model. [Audited outputs](../../results/nonlinear_retrieval_natural/iirc_reader_v1/summary.json)

| Reader evidence | All, n=134 | Single span, n=90 | Yes/no, n=25 | Numeric, n=19 | Median prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Question only | 20 | 3 | 15 | 2 | 30.5 |
| Source lead | 21 | 16 | 4 | 1 | 152.5 |
| Source lead and gold linked article leads | 25 | 24 | 1 | 0 | 482 |
| Source lead and gold linked oracle windows | 26 | 25 | 1 | 0 | 482 |
| Annotated source and linked spans | **38** | **35** | 3 | 0 | 135.5 |

The annotation-informed linked windows added one net exact match over the leads, with two wins and one loss (**+0.75 points**, interval **−1.55 to +3.23**). Annotated spans made the evidence more focused, changed the original source text, and shortened the prompt. This comparison captures their joint effect. The exact gains are concentrated in single-span answers. On binary questions the question-only arm emitted a literal form of “no” on 22 of 25. With all three leads, 23 of 25 outputs differed from the literal yes/no forms checked by the audit. This is a format failure under the generic reader prompt. [Label and output-hash replay](../../results/nonlinear_retrieval_natural/iirc_reader_v1/)

## Matched inputs and scoring

We began from the 139 fully aligned development questions in the evidence-visibility audit. Five multi-span answers were removed before decoding, leaving 134 cases across 107 source articles. The reader was given the two correct linked article IDs in a fixed order whenever linked evidence was included. The per-section token limits were 120 for the source and 165 for each linked article, with all final prompts at most 512 tokens. Scope and answer types were fixed before running HPG array **52956455** on four L4 shards. [Visibility audit](IIRC_EVIDENCE_VISIBILITY.md) · [Slurm driver](../../slurm/nonlinear_retrieval_iirc_reader.sbatch)

For binary and single-span strings, exactness uses the repository's case, punctuation, whitespace, and article normalization. Numeric exactness compares a single parsed value, allowing the recorded unit as an optional suffix. The original IIRC study uses a different reader and evaluation regime. This run measures the pinned reader under fixed article identities and a generic prompt. The zero numeric matches on all three linked arms and the binary output formats leave substantial reader headroom even with good evidence. A type-aware format is a separate experiment that should be inferred from the question or evaluated explicitly as an oracle. [IIRC paper](https://aclanthology.org/2020.emnlp-main.86/)

## Reproduce

```bash
base=results/nonlinear_retrieval_natural
study="$base/iirc_reader_v1"
OPENBLAS_NUM_THREADS=1 python scripts/analyze_kam_iirc_reader.py "$study/replay.json.gz" \
  --visibility "$base/iirc_visibility_v1/replay.json.gz" --output /tmp/kam-iirc-reader.json
diff -u "$study/summary.json" /tmp/kam-iirc-reader.json
```

The [fixture builder](../../scripts/prepare_kam_iirc_reader.py) regenerates private prompts from the official archives and the numeric visibility record. The public replay records IDs, token counts, exact labels, and output hashes.
