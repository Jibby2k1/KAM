#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_mixed_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
platform="${2:?platform name required}"
work="$root/radial_mixed_long_training_v1"
archive="$root/radial_archive_training_v1"
export PYTHONPATH="$work" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
(cd "$archive" && sha256sum --quiet -c "$work/archive.sha256")
seeds=(931121 931157 931187)
python -u "$work/run_kam_radial_archive_training.py" --earlier "$archive/earlier.json" --later "$archive/later.json" \
    --curriculum mixed-long --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --order "$SLURM_ARRAY_TASK_ID" \
    --output "$work/train-$platform-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
