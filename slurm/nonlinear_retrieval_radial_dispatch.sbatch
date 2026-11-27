#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_dispatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:09:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
platform="${2:?platform name required}"
work="$root/radial_dispatch_v1"
export PYTHONPATH="$work" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(638982 639003 639051)
python -u "$work/run_kam_radial_dispatch.py" --checkpoint "$root/bitemporal_sdpa_v1/decoder.pt" \
    --checkpoint-sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f \
    --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --order "$SLURM_ARRAY_TASK_ID" \
    --output "$work/result-$platform-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
