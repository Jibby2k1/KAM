#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_atomic_bindings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:06:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
stage="${2:?fit or search required}"
work="$root/native_musique_atomic_bindings_v1"
native="$root/native_musique_v1"
data="$root/native_musique_decomposition_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
seeds=(106331 106339 106367)
case "$stage" in
  fit)
    weights="$work/atomic-$SLURM_ARRAY_JOB_ID-$i.pt"
    python -u "$work/train_kam_musique_decomposer.py" fit --assets "$root/assets" --data "$data/data.json.gz" \
        --manifest "$data/manifest.json.gz" --seed "${seeds[$i]}" --binding-tokens --weights "$weights" \
        --output "$work/fit-$SLURM_ARRAY_JOB_ID-$i.json"
    for mode in greedy masked; do
        options=()
        if [[ $mode = masked ]]; then options=(--enforce-binding-order); fi
        python -u "$work/train_kam_musique_decomposer.py" inference --assets "$root/assets" \
            --fixture "$native/fixture.json" --weights "$weights" "${options[@]}" \
            --output "$work/plan-$SLURM_ARRAY_JOB_ID-$i-$mode.json.gz"
    done
    ;;
  search)
    fit="${3:?fit job ID required}"
    baseline="${4:?frozen original search job ID required}"
    for k in 0 1 2; do
        for mode in greedy masked; do
            python -u "$work/run_kam_musique_learned_search.py" --assets "$root/assets" --fixture "$native/fixture.json" \
                --corpus "$native/corpus.jsonl.gz" --vectors "$native"/embedding-[0-3].pt \
                --generated "$work/plan-$fit-$k-$mode.json.gz" --baseline "$native/search-$baseline-$i.json.gz" \
                --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" \
                --output "$work/rank-$SLURM_ARRAY_JOB_ID-$i-$k-$mode.json.gz"
        done
    done
    ;;
  *) exit 2 ;;
esac
