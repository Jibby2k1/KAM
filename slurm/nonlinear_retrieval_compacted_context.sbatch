#!/usr/bin/env bash
#SBATCH --job-name=kam_compacted_context
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:02:30
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
device="${2:?hardware label required}"
iterations="${3:-24}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work" OMP_NUM_THREADS=4
arms=(dot_without_absolute radial_without_absolute)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/benchmark_kam_compacted_context.py" --arm "${arms[$i]}" --steps "$iterations" --output "$work/$device-$i.json"
