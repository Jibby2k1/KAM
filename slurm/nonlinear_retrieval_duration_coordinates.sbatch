#!/usr/bin/env bash
#SBATCH --job-name=kam_duration_coordinates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/duration_coordinates_v1"
export PYTHONPATH="$work" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(542981 543001 543037)
python -u "$work/run_kam_duration_coordinates.py" --fixture "$root/unseentimeqa_duration_swaps_v1/fixture.json" \
    --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --order "$SLURM_ARRAY_TASK_ID" \
    --output "$work/coordinates-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
