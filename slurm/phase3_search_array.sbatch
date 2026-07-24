#!/usr/bin/env bash
#SBATCH --job-name=kam_p3_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${HPG_REPO:?HPG_REPO is required}"
: "${HPG_ENV:?HPG_ENV is required}"
module purge
module load cuda/12.8.1
source "$HPG_ENV/bin/activate"
cd "$HPG_REPO"
python -m kam.phase3.run_array \
  --manifest "${PHASE3_MANIFEST:?PHASE3_MANIFEST is required}" \
  --array-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --run-root "${PHASE3_RUN_ROOT:?PHASE3_RUN_ROOT is required}" \
  --device auto --resume
