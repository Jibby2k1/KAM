#!/usr/bin/env bash
#SBATCH --job-name=kam_p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

: "${HPG_ACCOUNT:?Set HPG_ACCOUNT}"
: "${HPG_QOS:?Set HPG_QOS}"
: "${HPG_PARTITION:?Set HPG_PARTITION at submission}"
: "${HPG_BLUE_ROOT:?Set HPG_BLUE_ROOT under /blue}"
: "${PHASE3_MANIFEST:?Set PHASE3_MANIFEST}"

ROW_ID="${SLURM_ARRAY_TASK_ID:?This script must run as an array task}"
RUN_ROOT="${HPG_BLUE_ROOT}/results/phase3"
mkdir -p "${RUN_ROOT}" logs

# Load the project environment here. Codex must replace this placeholder with
# the repository's validated module/conda/container activation.
# module load ...
# source .../bin/activate

python -m kam.phase3 run-row \
  --manifest "${PHASE3_MANIFEST}" \
  --row-id "${ROW_ID}" \
  --run-root "${RUN_ROOT}"
