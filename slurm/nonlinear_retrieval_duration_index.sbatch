#!/usr/bin/env bash
#SBATCH --job-name=kam_duration_index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/duration_index_v1"
export PYTHONPATH="$work" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
srun --cpu-bind=cores python -u "$work/benchmark_kam_duration_index.py" \
    --fixture "$root/unseentimeqa_duration_swaps_v1/fixture.json" --output "$work/index-$SLURM_JOB_ID.json"
