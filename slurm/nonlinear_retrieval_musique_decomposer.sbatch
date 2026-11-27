#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_decomposer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:18:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?fit or inference required}"
work="$root/native_musique_decomposition_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
runner=(python -u "$work/train_kam_musique_decomposer.py" "$mode" --assets "$root/assets")
seeds=(106331 106339 106367)
case "$mode" in
    fit)
        i="$SLURM_ARRAY_TASK_ID"
        "${runner[@]}" --data "$work/data.json.gz" --manifest "$work/manifest.json.gz" --seed "${seeds[$i]}" \
            --weights "$work/decomposer-$SLURM_ARRAY_JOB_ID-$i.pt" --output "$work/train-$SLURM_ARRAY_JOB_ID-$i.json"
        ;;
    inference)
        training="${3:?training job ID or baseline required}"
        i="$SLURM_ARRAY_TASK_ID"
        if [[ "$training" == baseline ]]; then
            weights=()
        else
            weights=(--weights "$work/decomposer-$training-$i.pt")
        fi
        "${runner[@]}" --fixture "$native/fixture.json" "${weights[@]}" \
            --output "$work/decompose-$training-$SLURM_ARRAY_JOB_ID-$i.json.gz"
        ;;
    *) exit 2 ;;
esac
