#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_atlas_repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l4:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${PHASE6_ATLAS_ENV:?PHASE6_ATLAS_ENV is required}/bin/activate"
cd "${PHASE6_ATLAS_REPO:?PHASE6_ATLAS_REPO is required}"
export PYTHONPATH="${PHASE6_ATLAS_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -u -m kam.phase6.behavioral_atlas_repair run \
  --manifest "${PHASE6_ATLAS_REPAIR_MANIFEST:?PHASE6_ATLAS_REPAIR_MANIFEST is required}" \
  --index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --stage0-run-root "${PHASE6_ATLAS_STAGE0_ROOT:?PHASE6_ATLAS_STAGE0_ROOT is required}" \
  --output-root "${PHASE6_ATLAS_REPAIR_ROOT:?PHASE6_ATLAS_REPAIR_ROOT is required}" \
  --device cuda
