#!/usr/bin/env bash
#SBATCH --job-name=kam_p3_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${HPG_BLUE_ROOT:?Set HPG_BLUE_ROOT under /blue}"

python -m kam.phase3 aggregate --run-root "${HPG_BLUE_ROOT}/results/phase3"
python -m kam.phase3 gate \
  --gate-config configs/phase3/gates.yaml \
  --run-root "${HPG_BLUE_ROOT}/results/phase3"
