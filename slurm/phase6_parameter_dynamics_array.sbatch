#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_pd_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l4:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${PHASE6_PD_ENV:?PHASE6_PD_ENV is required}/bin/activate"
cd "${PHASE6_PD_REPO:?PHASE6_PD_REPO is required}"
export PYTHONPATH="${PHASE6_PD_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -u -m kam.phase6.parameter_dynamics_runner \
  --manifest "${PHASE6_PD_MANIFEST:?PHASE6_PD_MANIFEST is required}" \
  --index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --output-root "${PHASE6_PD_RUN_ROOT:?PHASE6_PD_RUN_ROOT is required}" \
  --device cuda
