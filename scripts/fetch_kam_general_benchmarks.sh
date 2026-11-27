#!/bin/bash
set -euo pipefail
data=$1
mkdir -p "$data"
fetch() {
    local name=$1 url=$2 sha=$3
    curl -fsSL --retry 3 "$url" -o "$data/$name.tmp"
    python3 - "$data/$name.tmp" "$sha" <<'PY'
import hashlib
import sys
with open(sys.argv[1], 'rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"SHA-256 mismatch for {sys.argv[1]}: {actual}")
PY
    mv "$data/$name.tmp" "$data/$name"
}
hf=https://huggingface.co/datasets
fetch humaneval.parquet \
    "$hf/openai/openai_humaneval/resolve/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet" \
    2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6
fetch mbpp.parquet \
    "$hf/google-research-datasets/mbpp/resolve/4bb6404fdc6cacfda99d4ac4205087b89d32030c/full/test-00000-of-00001.parquet" \
    566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77
fetch mmlu.parquet \
    "$hf/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet" \
    74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7
fetch gsm8k.parquet \
    "$hf/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet" \
    ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59
fetch gpqa.csv https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv \
    41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305
