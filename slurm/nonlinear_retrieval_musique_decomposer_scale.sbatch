#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_decomposer_scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?fit or search required}"
work="$root/native_musique_decomposer_scale_v1"
native="$root/native_musique_v1"
data="$root/native_musique_decomposition_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
seeds=(106331 106339 106367)
case "$mode" in
  fit)
    sizes=(small base)
    if (( i % 2 )); then sizes=(base small); fi
    for size in "${sizes[@]}"; do
        weights="$work/decomposer-$SLURM_ARRAY_JOB_ID-$i-$size.pt"
        python -u "$work/train_kam_musique_decomposer.py" fit --assets "$root/assets" --data "$data/data.json.gz" \
            --manifest "$data/manifest.json.gz" --seed "${seeds[$i]}" --model-size "$size" --weights "$weights" \
            --output "$work/fit-$SLURM_ARRAY_JOB_ID-$i-$size.json"
        python -u "$work/train_kam_musique_decomposer.py" inference --assets "$root/assets" \
            --fixture "$native/fixture.json" --weights "$weights" --output "$work/plan-$SLURM_ARRAY_JOB_ID-$i-$size.json.gz"
    done
    ;;
  search)
    fit="${3:?fit array ID required}"
    baseline="${4:?baseline array ID required}"
    for seed_index in 0 1 2; do
        for size in small base; do
            python -u "$work/run_kam_musique_learned_search.py" --assets "$root/assets" --fixture "$native/fixture.json" \
                --corpus "$native/corpus.jsonl.gz" --vectors "$native"/embedding-[0-3].pt \
                --generated "$work/plan-$fit-$seed_index-$size.json.gz" --baseline "$native/search-$baseline-$i.json.gz" \
                --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" \
                --output "$work/rank-$SLURM_ARRAY_JOB_ID-$i-$seed_index-$size.json.gz"
        done
    done
    ;;
  *) exit 2 ;;
esac
