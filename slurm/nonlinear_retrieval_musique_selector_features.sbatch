#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_selector_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=14G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?fit or test required}"
work="$root/native_musique_selector_features_v1"
fit="$root/native_musique_selector_v1"
native="$root/native_musique_v1"
learned="$root/native_musique_learned_search_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
runner=(python -u "$work/run_kam_musique_selector_features.py" --assets "$root/assets" --corpus "$native/corpus.jsonl.gz"
        --vectors "$native"/embedding-[0-3].pt)
case "$mode" in
  fit)
    "${runner[@]}" --fixture "$fit/fixture.json" --baseline "$fit/baseline-${3:?baseline job ID required}.json.gz" \
        --learned "$fit"/rank-"${4:?learned job ID required}"-[0-2].json.gz --shard 0 --shards 1 \
        --output "$work/features-fit-$SLURM_JOB_ID.json.gz"
    ;;
  test)
    prior="${3:?baseline job ID required}"
    shift 3
    ranks=()
    for job in "$@"; do ranks+=("$learned/learned-$job-$SLURM_ARRAY_TASK_ID.json.gz"); done
    "${runner[@]}" --fixture "$native/fixture.json" --baseline "$native/search-$prior-$SLURM_ARRAY_TASK_ID.json.gz" \
        --learned "${ranks[@]}" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
        --output "$work/features-test-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
    ;;
  *) exit 2 ;;
esac
