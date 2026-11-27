#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_selector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?baseline, plan, or rank required}"
work="$root/native_musique_selector_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
case "$mode" in
  baseline)
    python -u "$native/run_kam_native_musique.py" search --corpus "$native/corpus.jsonl.gz" --fixture "$work/fixture.json" \
        --assets "$root/assets" --lexical "$native/lexical" --vectors "$native"/embedding-[0-3].pt \
        --output "$work/baseline-$SLURM_JOB_ID.json.gz"
    ;;
  plan)
    trained="${3:?training job ID required}"
    i="$SLURM_ARRAY_TASK_ID"
    python -u "$work/train_kam_musique_decomposer.py" inference --assets "$root/assets" --fixture "$work/fixture.json" \
        --trained-for 0d6a4cb740d63047f0ba03b4a06f7c1dc7563852bb37cc50e9138fd3493b72dc \
        --weights "$root/native_musique_decomposition_v1/decomposer-$trained-$i.pt" \
        --output "$work/plan-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  rank)
    initial="${3:?selector fit baseline job ID required}"
    predicted="${4:?selector fit plan array job ID required}"
    i="$SLURM_ARRAY_TASK_ID"
    python -u "$work/run_kam_musique_learned_search.py" --assets "$root/assets" --corpus "$native/corpus.jsonl.gz" \
        --fixture "$work/fixture.json" --vectors "$native"/embedding-[0-3].pt --shard 0 --shards 1 \
        --baseline "$work/baseline-$initial.json.gz" --generated "$work/plan-$predicted-$i.json.gz" \
        --output "$work/rank-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  *) exit 2 ;;
esac
