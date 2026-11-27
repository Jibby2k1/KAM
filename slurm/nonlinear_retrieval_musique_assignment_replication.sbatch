#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_assignment_replication
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?search, plans or read required}"
work="$root/native_musique_assignment_replication_v1"
native="$root/native_musique_v1"
decomp="$root/native_musique_decomposition_v1"
export PYTHONPATH="$work:$decomp:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
n="$SLURM_ARRAY_TASK_COUNT"
common=(--assets "$root/assets" --corpus "$native/corpus.jsonl.gz" --fixture "$work/fixture.json" --vectors "$native"/embedding-[0-3].pt)
case "$mode" in
  search)
    python -u "$native/run_kam_native_musique.py" search "${common[@]}" --lexical "$native/lexical" \
        --shard "$i" --shards "$n" --output "$work/search-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  plans)
    train="${3:?decomposer training job ID required}"
    source="${4:?original fixture hash required}"
    python -u "$work/train_kam_musique_decomposer.py" inference --assets "$root/assets" \
        --fixture "$work/fixture.json" --weights "$decomp/decomposer-$train-$i.pt" --trained-for "$source" \
        --output "$work/plan-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  read)
    search="${3:?baseline search job ID required}"
    plans="${4:?plan inference job ID required}"
    rank=()
    for seed in 0 1 2; do
        file="$work/affinity-$SLURM_ARRAY_JOB_ID-$i-$seed.json.gz"
        python -u "$work/run_kam_musique_learned_search.py" "${common[@]}" --shard "$i" --shards "$n" \
            --baseline "$work/search-$search-$i.json.gz" --generated "$work/plan-$plans-$seed.json.gz" \
            --emit-affinity --output "$file"
        rank+=("$file")
    done
    python -u "$work/run_kam_musique_final_reader.py" run "${common[@]}" --labels "$work/answers.json" \
        --baseline "$work/search-$search-$i.json.gz" --learned "${rank[@]}" --joint-assignment --shard "$i" --shards "$n" \
        --output "$work/read-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  *) exit 2 ;;
esac
