#!/usr/bin/env bash
#SBATCH --job-name=kam_bright_windows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?encode or search required}"
work="$root/${KAM_BRIGHT_WINDOWS_STAGE:-bright_passage_windows_v1}"
runtime="$root/bright_transfer_v1"
data="$root/${KAM_BRIGHT_DATA_STAGE:-bright_transfer_v1}"
export PYTHONPATH="$work:$data:$runtime:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
read -r -a domains <<< "${KAM_BRIGHT_DOMAIN_PAIR:-economics earth_science}"
[[ "${#domains[@]}" -eq 2 ]]
case "$mode" in
    encode|encode_both)
        domain="${domains[$((i / 4))]}"
        shard="$((i % 4))"
        if [[ "$mode" == "encode_both" ]]; then
            python -u "$runtime/run_kam_bright_transfer.py" encode --assets "$root/assets" --inputs "$data/$domain" \
                --shard "$shard" --output "$data/$domain/key-$SLURM_ARRAY_JOB_ID-$shard.pt"
        fi
        python -u "$work/run_kam_bright_passage_windows.py" encode --assets "$root/assets" --inputs "$data/$domain" \
            --shard "$shard" --output "$work/$domain/windows-$SLURM_ARRAY_JOB_ID-$shard.pt"
        ;;
    bm25)
        domain="${domains[$i]}"
        python -u "$runtime/run_kam_bright_bm25.py" --inputs "$data/$domain" --output "$data/$domain/bm25.json.gz"
        ;;
    search)
        domain="${domains[$i]}"
        base="${3:?prior embedding array required}"
        windows="${4:?window embedding array required}"
        shift 4
        vectors="$root/${KAM_BRIGHT_VECTOR_STAGE:-${KAM_BRIGHT_WINDOWS_STAGE:-bright_passage_windows_v1}}"
        python -u "$work/run_kam_bright_passage_windows.py" search --assets "$root/assets" --inputs "$data/$domain" \
            --head "$data/$domain"/key-"$base"-[0-3].pt --extra "$vectors/$domain"/windows-"$windows"-[0-3].pt \
            --lexical "$data/$domain/bm25.json.gz" --output "$work/$domain/rank-$SLURM_ARRAY_JOB_ID-$i.json.gz" "$@"
        ;;
    *) exit 2 ;;
esac
