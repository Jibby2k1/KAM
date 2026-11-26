#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_factor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
platform="${2:?platform label required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_factorization.py" --output "$work/$platform.json"
