#!/usr/bin/env bash
#SBATCH --job-name=kam_p3_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
: "${HPG_REPO:?HPG_REPO is required}"
: "${HPG_ENV:?HPG_ENV is required}"
source "$HPG_ENV/bin/activate"
cd "$HPG_REPO"
python -m kam.phase3.aggregate \
  --run-root "${PHASE3_RUN_ROOT:?PHASE3_RUN_ROOT is required}" \
  --output "${PHASE3_RUN_ROOT:?PHASE3_RUN_ROOT is required}/phase3_aggregate.json" \
  --report-root "$HPG_REPO/reports/phase3"
