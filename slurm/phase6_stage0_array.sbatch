#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_s0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${PHASE6_ENV:?PHASE6_ENV is required}/bin/activate"
cd "${PHASE6_REPO:?PHASE6_REPO is required}"
python -u -m kam.phase6.run_stage0 --manifest "${PHASE6_MANIFEST:?PHASE6_MANIFEST is required}" --output "${PHASE6_RUN_ROOT:?PHASE6_RUN_ROOT is required}/row_${SLURM_ARRAY_TASK_ID}.jsonl" --row-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
