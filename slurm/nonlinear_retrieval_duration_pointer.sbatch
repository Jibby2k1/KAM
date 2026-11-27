#!/usr/bin/env bash
#SBATCH --job-name=kam_duration_pointer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/duration_pointer_v1"
export PYTHONPATH="$work" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(452671 452701 452731)
python -u "$work/run_kam_duration_pointer.py" --fixture "$root/unseentimeqa_duration_swaps_v1/fixture.json" \
    --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --order "$SLURM_ARRAY_TASK_ID" \
    --output "$work/pointer-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
