# Reproduce the dataset and temporal memory studies

The [research handoff](../../reports/nonlinear_retrieval_natural/DATASET_BREADTH_2026-11-26.md) links each analysis to a compact result. The [source manifest](../../results/nonlinear_retrieval_natural/dataset_breadth_2026-11-26/sources.json) records input files, byte hashes, prepared fixtures, and frozen model revisions. The data-preparation scripts build selected cohorts from the upstream releases. The public repository supplies raw run-level evidence for the controlled studies and summaries for the larger natural-text studies.

## Audit checked-in results on a CPU

Run these commands from the repository root with Python 3.11 or newer. The aggregator uses only the standard library.

```bash
archive=results/nonlinear_retrieval_natural
scratch=$(mktemp -d)
for pair in corrections:bitemporal_corrections_v1 curriculum:bitemporal_curriculum_v1 \
            archive_projection:bitemporal_archive_v1 archive_topology:bitemporal_archive_topology_v1 \
            archive_factors:bitemporal_archive_factors_v1 vintage_stream:vintage_stream_v1 \
            timelineqa:timelineqa_v1 unseentimeqa_probe:unseentimeqa_probe_v1; do
  analysis=${pair%%:*}
  study="$archive/${pair#*:}"
  python scripts/analyze_kam_breadth_followups.py "$analysis" "$study" --output "$scratch/$analysis.json"
  diff -u "$study/summary.json" "$scratch/$analysis.json" || exit 1
done
```

## Regenerate the synthetic decoder experiment

Install the repository with a CUDA-enabled PyTorch build. HPG used PyTorch 2.8.0. The decoder learns from generated typed inputs. The following reference arm uses the same seed, updates, curriculum, and clock fields as one of the four seed-matched runs.

```bash
mkdir -p outputs/dataset-breadth
python scripts/run_kam_bitemporal_decoder.py \
  --arm radial_without_absolute --schedule wide --no-query-clock --extended \
  --seed 343733 --steps 3200 --eval-examples 512 \
  --output outputs/dataset-breadth/radial-wide.json
```

Use the arrays in the [curriculum driver](../../slurm/nonlinear_retrieval_bitemporal_curriculum.sbatch) and [same-date correction driver](../../slurm/nonlinear_retrieval_bitemporal_corrections.sbatch) for the complete comparisons. The array scripts assume a staged directory containing the scripts and dependencies and a standard `sha256sum -c` manifest. HPG-specific module and accelerator settings can be replaced by an equivalent local environment. Reuse authoritative sharding and seed lists when changing systems.

The [FlashAttention driver](../../slurm/nonlinear_retrieval_radial_sdpa.sbatch) fits the same checkpoint once, then evaluates it on both GPU families. Use `--checkpoint` with the decoder to write a fresh state and `--cuda-graph` with `benchmark_kam_radial_sdpa.py` to include fixed-shape replay. The benchmark validates outputs against the FP32 reference and checks that graph replay can accept a second batch through the captured input buffer.

## Recreate real revision lookup

Download `fed-ROUTPUT.xlsx`, `fed-NOUTPUT.xlsx`, `fed-RCON.xlsx`, and `fed-NCON.xlsx` from the URLs under `vintage` in the source manifest. Save them with the listed names. Compare file hashes before fixture creation because these official archives receive updates. The preparation script independently compares its labels with SQLite.

```bash
mkdir -p outputs/dataset-breadth
python scripts/prepare_kam_vintage_revisions.py \
  --source-directory outputs/dataset-breadth \
  --output outputs/dataset-breadth/fed-vintage-fixture.json \
  --catalog-output outputs/dataset-breadth/fed-vintage-catalog.json
python scripts/run_kam_vintage_stream.py \
  --catalog outputs/dataset-breadth/fed-vintage-catalog.json \
  --seed 230619 --queries 200000 --output outputs/dataset-breadth/stream-230619.json
```

For GPU tests on raw queries, install the dependencies and retrieve the frozen models as in the [HotpotQA guide](REPRODUCE_HOTPOTQA.md#prepare-the-corpus-and-models). `run_kam_vintage_global.py` takes an exact corpus, prepared fixture, and encoder assets. `run_kam_vintage_revisions.py` compares alias, eligibility, and selection on the prepared pools. Each driver has `--help` and a Slurm counterpart with the settings used in the study. Replayed timing depends on input sizes, numerical backend, CPU, GPU, and whether data is resident. Compare accuracy and rank parity independently of timing.

Use `--archive-fixture` with `run_kam_bitemporal_decoder.py` to score typed archive histories. The [archive driver](../../slurm/nonlinear_retrieval_bitemporal_archive.sbatch) matches training tokens and switches among original procedural data, synthetic corrections, and earlier workbook groups. Use the [boundary driver](../../slurm/nonlinear_retrieval_bitemporal_boundary.sbatch) for the label-preserving clock ablation.

For the release-cluster comparison, regenerate the archive fixture with `--test-years 2023 2026`. The interval is half-open, with enough verified candidates to retain observations from 2023 and 2024. Use the [release-cluster driver](../../slurm/nonlinear_retrieval_bitemporal_archive_topology.sbatch) for the independently held-out comparison and the [factor driver](../../slurm/nonlinear_retrieval_bitemporal_archive_factors.sbatch) to study the earlier cohort.

## Audit the serial logistics labels

Check out the revision in the source manifest, then use the upstream `data` directory. The parser runs on a CPU with the Python standard library.

```bash
python scripts/audit_kam_unseentimeqa.py \
  --root /path/to/UnSeenTimeQA/data \
  --revision 84f910fcd32f2b21d6e34d9efb98e94eb04844bb \
  --output outputs/dataset-breadth/unseentimeqa.json
diff -u results/nonlinear_retrieval_natural/unseentimeqa_v1/summary.json outputs/dataset-breadth/unseentimeqa.json
```

The audit parses the serial easy and medium event texts, derives a package availability trace, and checks an independently sampled time offset on every item. The released set-valued label is used only for evaluation.

Use `scripts/prepare_kam_unseentimeqa_probe.py --root /path/to/UnSeenTimeQA/data --output outputs/dataset-breadth/unseentimeqa-probe-fixture.json` to generate the retrieval panel. The [probe driver](../../slurm/nonlinear_retrieval_unseentimeqa_probe.sbatch) uses four GPU shards and the same E5 encoder as the lifelog probe. The scoped rule uses the fixture's known initial-world identity and the parsed event interval.

## Regenerate the lifelog provenance fixture

The [TimelineQA result](../../results/nonlinear_retrieval_natural/timelineqa_v1/summary.json) records all 24 generator seeds and the hashes of their source episode files. Check out the TimelineQA revision in the source manifest. Generate from the upstream `src` directory, using one output directory per seed.

```bash
archive="$PWD/results/nonlinear_retrieval_natural"
timeline="$PWD/outputs/dataset-breadth/timeline"
upstream=/path/to/TimelineQA
mkdir -p "$timeline"
python - "$archive/timelineqa_v1/summary.json" > "$timeline/personas.tsv" <<'PY'
import json
import sys

for row in json.load(open(sys.argv[1]))["sources"]:
    print(row["seed"], row["density"])
PY
while read -r seed density; do
  mkdir -p "$timeline/$seed"
  (cd "$upstream/src" && PYTHONHASHSEED=0 python generateDB.py \
    -y 2022 -s "$seed" -c "$density" -d "$timeline/$seed" -o episodes.json)
done < "$timeline/personas.tsv"
python scripts/prepare_kam_timelineqa.py --episodes "$timeline" \
  --manifest "$timeline/personas.tsv" --generator-revision e0ff14f707ce07b1f656e61b2921059c3fc98168 \
  --output "$timeline/timeline-fixture.json"
```

Run `scripts/run_kam_timelineqa.py` on eight independent shards with the pinned E5 encoder. The public raw result files allow CPU replay of the aggregate with `analyze_kam_breadth_followups.py timelineqa`.

## Data lineage

Most source corpora are obtained directly from the dataset authors or registered dataset hosts. The archive and narrative reports use compact derived results. The external corpora, workbook binaries, raw conversation text, and generated narratives are acquired or regenerated upstream. The full task names and citations are in the [dataset portfolio](DATASET_PORTFOLIO.md).

## Audit later cutoff, passage-window and precision controls

The numeric analyses run without a GPU. From the repository root, with NumPy installed, use the immutable result records.

```bash
archive=results/nonlinear_retrieval_natural
scratch=$(mktemp -d)
study="$archive/clock_release_mixture_v1"
python scripts/analyze_kam_clock_release_training.py \
  --unemployment "$archive/eurostat_vintages_v1/fixture.json.gz" \
  --industry "$archive/eurostat_industry_v1/fixture.json.gz" \
  --later-unemployment "$study/later-unemployment.json.gz" \
  --later-industry "$study/later-industry.json.gz" \
  --measurement "$study"/train-53003489-*.json.gz --output "$scratch/cutoff.json"
diff -u "$study/summary-53003489.json" "$scratch/cutoff.json"

study="$archive/bright_passage_windows_v2/psychology"
python scripts/analyze_kam_bright_passage_windows.py --manifest "$study/manifest.json" \
  --measurement "$study/rank-53003539-0.json.gz" --reference "$study/rank-53003100-0.json.gz" \
  --output "$scratch/bright.json"
diff -u "$study/summary.json" "$scratch/bright.json"

study="$archive/precision_margin_fallback_v1"
python scripts/analyze_kam_precision_margin_fallback.py --measurement "$study"/timing-*.json.gz \
  --output "$scratch/precision.json"
diff -u "$study/summary.json" "$scratch/precision.json"
```

Use the [Eurostat fixture builder](../../scripts/prepare_kam_eurostat_vintages.py) with the indicator and cutoff settings in each manifest to verify archived labels against SQLite. The [release-phase driver](../../slurm/nonlinear_retrieval_clock_release_training.sbatch) uses the same synthetic banks and update order as the original Eurostat replay. `--compare-mixtures` activates fixed seeded mixtures. The [passage window driver](../../slurm/nonlinear_retrieval_bright_passage_windows.sbatch) shares the original BRIGHT source revision, per-domain shortlists and frozen E5 bank. Its `--sham` search option reassigns additional vectors among similar-length documents.
