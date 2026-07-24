#!/usr/bin/env bash
#SBATCH --job-name=kam_p5_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${HPG_ENV:?HPG_ENV is required}/bin/activate"
cd "${HPG_REPO:?HPG_REPO is required}"
python -m kam.phase5.pilot_run --manifest "${PHASE5_MANIFEST:?PHASE5_MANIFEST is required}" --array-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" --run-root "${PHASE5_RUN_ROOT:?PHASE5_RUN_ROOT is required}" --device auto --resume
