#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_scale_readout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?fresh_plans or read required}"
evaluation="${3:?native or fresh required}"
fit="${4:?fit job ID required}"
work="$root/native_musique_scale_readout_v1"
training="$root/native_musique_decomposer_scale_v1"
native="$root/native_musique_v1"
if [[ "$evaluation" == native ]]; then
    dataset="$native"
    labels="$root/native_musique_final_reader_v1/answers.json"
else
    dataset="$root/native_musique_assignment_replication_v1"
    labels="$dataset/answers.json"
fi
export PYTHONPATH="$work:$training:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
n="$SLURM_ARRAY_TASK_COUNT"
common=(--assets "$root/assets" --corpus "$native/corpus.jsonl.gz" --fixture "$dataset/fixture.json" --vectors "$native"/embedding-[0-3].pt)
case "$mode" in
    fresh_plans)
        source="$5"
        for size in small base; do
            python -u "$training/train_kam_musique_decomposer.py" inference --assets "$root/assets" \
                --fixture "$dataset/fixture.json" --weights "$training/decomposer-$fit-$i-$size.pt" --trained-for "$source" \
                --output "$work/plan-$SLURM_ARRAY_JOB_ID-$i-$size.json.gz"
        done
        ;;
    read)
        search="${5:?baseline search job ID required}"
        plans="${6:?plan job ID required}"
        plan_dir="$work"
        if [[ "$evaluation" == native ]]; then plan_dir="$training"; fi
        baseline="$dataset/search-$search-$i.json.gz"
        affinity=()
        for seed_index in 0 1 2; do
            for size in small base; do
                file="$work/affinity-$SLURM_ARRAY_JOB_ID-$i-$seed_index-$size.json.gz"
                python -u "$work/run_kam_musique_learned_search.py" "${common[@]}" --shard "$i" --shards "$n" \
                    --baseline "$baseline" --generated "$plan_dir/plan-$plans-$seed_index-$size.json.gz" \
                    --emit-affinity --output "$file"
                affinity+=("$file")
            done
        done
        python -u "$work/run_kam_musique_final_reader.py" run "${common[@]}" --labels "$labels" \
            --baseline "$baseline" --learned "${affinity[@]}" --joint-assignment --shard "$i" --shards "$n" \
            --output "$work/read-$SLURM_ARRAY_JOB_ID-$i.json.gz"
        ;;
    *) exit 2 ;;
esac
