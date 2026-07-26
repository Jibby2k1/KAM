#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_night_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l4:1
set -euo pipefail

module purge
module load cuda/12.8.1
source "${PHASE6_ENV:?PHASE6_ENV is required}/bin/activate"
cd "${PHASE6_REPO:?PHASE6_REPO is required}"
export PYTHONPATH="${PHASE6_REPO}:${PYTHONPATH:-}"

python -u -m kam.phase6.overnight_runner \
  --manifest "${PHASE6_MANIFEST:?PHASE6_MANIFEST is required}" \
  --index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --output-root "${PHASE6_RUN_ROOT:?PHASE6_RUN_ROOT is required}" \
  --calibration "${PHASE6_RUN_ROOT}/calibration.json" \
  --device cuda
