#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_boundary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work"
arms=(dot_without_absolute radial_without_absolute)
sources=(procedural procedural_corrections procedural_boundary procedural_dual_boundary)
seeds=(4501459 4913063 5219177 5667173)
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 8))]}"
arm="${arms[$((SLURM_ARRAY_TASK_ID / 4 % 2))]}"
source="${sources[$((SLURM_ARRAY_TASK_ID % 4))]}"
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_decoder.py" --seed "$seed" --arm "$arm" \
  --steps 3200 --bank 8192 --eval-examples 256 --no-query-clock --training-data "$source" \
  --archive-boundary-diagnostic --archive-fixture "$work/archive-fixture.json" \
  --output "$work/eval-$seed-$arm-$source.json"
