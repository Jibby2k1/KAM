#!/usr/bin/env bash
#SBATCH --job-name=kam_p5_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${STAGE2_ENV:?STAGE2_ENV is required}/bin/activate"
cd "${STAGE2_REPO:?STAGE2_REPO is required}"
python -u -m kam.phase5.stage2_run --manifest "${STAGE2_MANIFEST:?STAGE2_MANIFEST is required}" --array-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" --run-root "${STAGE2_RUN_ROOT:?STAGE2_RUN_ROOT is required}" --device auto --resume >/dev/null
