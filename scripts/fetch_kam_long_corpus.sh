#!/bin/bash
set -euo pipefail
root=$1
mkdir -p "$root/source" "$root/corpus" "$root/logs"
base=https://huggingface.co/datasets
wiki=$base/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1
code=$base/code-search-net/code_search_net/resolve/bd0cf261e357a3eb5c8fba490d23ec1a1cd59555/python
fetch() {
    local url=$1 file=$2
    curl -fsSL --retry 3 "$url" -o "$file.tmp"
    mv "$file.tmp" "$file"
}
pids=()
for split in train validation test; do
    count=00001
    if [[ "$split" == train ]]; then count=00002; fi
    fetch "$wiki/$split-00000-of-$count.parquet" "$root/source/wiki-$split.parquet" &
    pids+=("$!")
    fetch "$code/$split-00000-of-00001.parquet" "$root/source/code-$split.parquet" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
