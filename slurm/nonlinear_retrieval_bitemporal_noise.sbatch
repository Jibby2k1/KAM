#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(295167 795147 394117 438176 880249 941127 631547 919833)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_noise.py" --seed "$seed" --output "$work/seed-$seed.json"
