#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_perm_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l4:1
set -euo pipefail
module purge
module load cuda/12.8.1
source "${PHASE6_ATLAS_ENV:?PHASE6_ATLAS_ENV is required}/bin/activate"
cd "${PHASE6_ATLAS_REPO:?PHASE6_ATLAS_REPO is required}"
export PYTHONPATH="${PHASE6_ATLAS_REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python -u -m kam.phase6.behavioral_atlas_permutation_audit run \
  --manifest "${PHASE6_ATLAS_PERMUTATION_AUDIT_MANIFEST:?PHASE6_ATLAS_PERMUTATION_AUDIT_MANIFEST is required}" \
  --index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --stage1-root "${PHASE6_ATLAS_STAGE1_ROOT:?PHASE6_ATLAS_STAGE1_ROOT is required}" \
  --output-root "${PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT:?PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT is required}" \
  --device cuda
