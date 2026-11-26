#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_archive
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
sources=(procedural procedural_corrections archive)
seeds=(2500891 2857097 3244409 3824407)
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 6))]}"
arm="${arms[$((SLURM_ARRAY_TASK_ID / 3 % 2))]}"
source="${sources[$((SLURM_ARRAY_TASK_ID % 3))]}"
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_decoder.py" --seed "$seed" --arm "$arm" \
  --steps 3200 --bank 8192 --eval-examples 256 --no-query-clock --training-data "$source" \
  --archive-fixture "$work/archive-fixture.json" --output "$work/eval-$seed-$arm-$source.json"
