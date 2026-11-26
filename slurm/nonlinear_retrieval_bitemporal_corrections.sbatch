#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_corrections
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(846497 874147 969547 974239 990481 1129869)
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 2))]}"
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then regime=baseline; else regime=corrections; fi
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_noise.py" --seed "$seed" --correction-panel \
  --train-regime "$regime" --output "$work/$regime-$seed.json"
