#!/usr/bin/env bash
#SBATCH --job-name=kam_bright_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?encode, bm25 or search required}"
data="$root/bright_transfer_v1"
work="$root/${KAM_BRIGHT_STAGE:-bright_transfer_v1}"
export PYTHONPATH="$work:$root/native_musique_decomposition_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
domains=(economics earth_science)
case "$mode" in
    encode)
        domain="${domains[$((i / 4))]}"
        shard="$((i % 4))"
        side="$data/$domain"
        python -u "$work/run_kam_bright_transfer.py" encode --assets "$root/assets" --inputs "$side" \
            --shard "$shard" --output "$side/key-$SLURM_ARRAY_JOB_ID-$shard.pt"
        ;;
    bm25)
        domain="${domains[$i]}"
        python -u "$work/run_kam_bright_bm25.py" --inputs "$data/$domain" --output "$data/$domain/bm25.json.gz"
        ;;
    search)
        domain="${domains[$i]}"
        keys="${3:?embedding array required}"
        fit="${4:?MuSiQue training array required}"
        side="$data/$domain"
        source="$root/native_musique_decomposer_scale_v1"
        weights=()
        for seed in 0 1 2; do
            for size in small base; do weights+=("$source/decomposer-$fit-$seed-$size.pt"); done
        done
        shift 4
        python -u "$work/run_kam_bright_transfer.py" search --assets "$root/assets" --inputs "$side" \
            --parts "$side"/key-"$keys"-[0-3].pt --lexical "$side/bm25.json.gz" --weights "${weights[@]}" \
            --output "$side/rank-$SLURM_ARRAY_JOB_ID-$i.json.gz" "$@"
        ;;
    *) exit 2 ;;
esac
