#!/usr/bin/env bash
#SBATCH --job-name=kam_native_musique
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?lexical, embed or search required}"
work="$root/native_musique_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/native_twowiki_v1:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
runner=(python -u "$work/run_kam_native_musique.py" "$mode" --corpus "$work/corpus.jsonl.gz"
        --fixture "$work/fixture.json" --assets "$root/assets")
case "$mode" in
    lexical) "${runner[@]}" --output "$work/lexical" ;;
    embed)
        "${runner[@]}" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
            --output "$work/embedding-$SLURM_ARRAY_TASK_ID.pt"
        ;;
    search)
        "${runner[@]}" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
            --lexical "$work/lexical" --vectors "$work"/embedding-[0-3].pt \
            --output "$work/search-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
        ;;
    *) exit 2 ;;
esac
