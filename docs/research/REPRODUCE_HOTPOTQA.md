# Reproduce the HotpotQA supporting-link study

The [study report](../../reports/nonlinear_retrieval_natural/SUPPORT_LINKS.md) describes the fixed comparisons. The archived evaluation can be audited on a CPU using Python's standard library. A fresh GPU run also requires the corpus, the pinned pretrained models, and their derived index.

## Audit the published metrics

Run from the repository root.

```bash
study=results/nonlinear_retrieval_natural/support_link_v1
python scripts/analyze_kam_fullwiki_support_links.py "$study" --output /tmp/kam-support-links-summary.json
diff -u "$study/summary.json" /tmp/kam-support-links-summary.json
```

The analyzer checks the fixture digest, exact fit and evaluation IDs, score alignment, candidate uniqueness, held-out gold exclusion from fitting pools, and the common checkpoint across the four evaluation shards.

## Prepare the corpus and models

The GPU study used Linux, PyTorch 2.8.0 with CUDA 12.8, Python 3.13.5, NumPy 2.3.2, Transformers 5.14.0, PyArrow 25.0.0, Tantivy 0.27.0, and Hugging Face Hub 1.23.0. Install a CUDA-enabled PyTorch build first, then the index and model dependencies.

```bash
python -m pip install numpy==2.3.2 transformers==5.14.0 pyarrow==25.0.0 tantivy==0.27.0 huggingface_hub==1.23.0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
```

Retrieve the corpus using the [corpus manifest](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/manifest.json). The original byte hash is required for the row IDs in the published split. The corpus download is roughly 1 GB and the 30 FP32 embedding shards together require roughly 8 GB before index and model storage.

```bash
study="$PWD/results/nonlinear_retrieval_natural/support_link_v1"
archive="$PWD/results/nonlinear_retrieval_natural/fullwiki_native_v1"
runtime="$PWD/outputs/support-links"
core="$runtime/fullwiki_native_v1"
mkdir -p "$core" "$runtime/assets"
cp "$archive"/{manifest.json,excluded_titles.txt,queries.json} "$core/"
corpus_url=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$core/manifest.json")
curl -fL "$corpus_url" -o "$core/corpus.parquet"
printf '%s  %s\n' 23975eb1ce371778a38b49a4cc0ca23998b4e1979ee31d02b3032a3e0cd5d62e "$core/corpus.parquet" | sha256sum -c
python - "$runtime/assets" "$study/summary.json" <<'PY'
import json
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

assets = Path(sys.argv[1])
revisions = json.loads(Path(sys.argv[2]).read_text())["models"]
for folder, repo in (("encoder", "intfloat/e5-small-v2"), ("reader", "google/flan-t5-small")):
    snapshot_download(repo, revision=revisions[repo], local_dir=assets / folder)
(assets / "revisions.json").write_text(json.dumps(revisions) + "\n")
PY
```

Build the article store and encode shards 0 through 29. The encoder shards are independent and can run as a Slurm array. Keep `--parts 6 --batch 64` and use an NVIDIA L4 to match the corpus run as closely as possible.

```bash
python scripts/build_kam_fullwiki_lexical.py --inputs "$core" --output "$core/lexical" --threads 8
printf '%s  %s\n' 652e023b2c253926dd09bb5ab11a78d13a071ac1065d99ad872fa50019d849bb "$core/lexical/allowed.npy" | sha256sum -c
# Run each shard on a GPU. Set shard to its array index when using Slurm.
for shard in $(seq 0 29); do
  python scripts/run_kam_fullwiki_encode.py --assets "$runtime/assets" \
    --corpus "$core/corpus.parquet" --manifest "$core/manifest.json" \
    --output "$core" --parts 6 --batch 64 --shard "$shard"
done
```

The published trained-head checkpoint records the original index metadata hash. To replay it on a rebuilt index, first verify the corpus and allow-mask hashes above, then copy the [recorded index metadata](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/lexical.json) over the generated `$core/lexical/completed.json`. Dense search uses the mask and embeddings. The test also loads the Tantivy store through the common index implementation. A fresh fit can use its own generated index metadata throughout.

## Fit and evaluate

The baseline checkpoint is pinned by SHA-256. Use the released trained heads for evaluation only, or fit fresh heads with the same settings and split.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
baseline=95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833
cfg=configs/nonlinear_retrieval/natural/hotpot_fullwiki_native_v1.json
mkdir -p "$runtime/replay"
cp "$study/fixture.json" "$runtime/replay/"
args=(--inputs "$core" --assets "$runtime/assets" --fixture "$study/fixture.json"
      --config "$cfg" --weights "$study/baseline.pt" --baseline-sha256 "$baseline")
python scripts/run_kam_fullwiki_support_links.py fit "${args[@]}" \
  --checkpoint "$runtime/replay/heads.pt" --output "$runtime/replay/fit-0.json"
# Submit the four iterations independently on a cluster.
for shard in 0 1 2 3; do
  python scripts/run_kam_fullwiki_support_links.py eval "${args[@]}" \
    --checkpoint "$runtime/replay/heads.pt" --output "$runtime/replay/eval-$shard.json" \
    --shard "$shard" --shards 4
done
python scripts/analyze_kam_fullwiki_support_links.py "$runtime/replay"
```

For evaluation only, set the checkpoint argument to `"$study/heads.pt"` and copy `fit-0.json` from the archived study to the replay directory. To match the reported GPU runtime, the [HPG Slurm script](../../slurm/nonlinear_retrieval_fullwiki_support_links.sbatch) launches the four evaluation shards with an array and checks the staged files. Use `KAM_ENV_SETUP` for a site-specific environment activation script and `KAM_BASELINE_SHA256` for the released checkpoint. Exact GPU regeneration can be sensitive to numerical ties, library versions, and hardware. The archived CPU audit is the reference for the published counts.

## Data attribution

The archived fixtures select HotpotQA bridge questions, retain the relevant source fields, and add document row IDs. They are derived from [HotpotQA by Yang et al. (EMNLP 2018)](https://aclanthology.org/D18-1259/) and the [BEIR HotpotQA corpus](https://huggingface.co/datasets/BeIR/hotpotqa), both distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The same license applies to the redistributed dataset fields and titles. The corpus and pretrained models are fetched from their respective upstream sources.
