#!/usr/bin/env bash
#SBATCH --job-name=kam_p4_report
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
python -m kam.phase4.aggregate \
  --run-root "${PHASE4_RUN_ROOT:?PHASE4_RUN_ROOT is required}" \
  --report-root "$HPG_REPO/reports/phase4"
