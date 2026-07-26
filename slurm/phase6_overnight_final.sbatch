#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_night_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail

source "${PHASE6_ENV:?PHASE6_ENV is required}/bin/activate"
cd "${PHASE6_REPO:?PHASE6_REPO is required}"
export PYTHONPATH="${PHASE6_REPO}:${PYTHONPATH:-}"

python -u scripts/build_phase6_overnight_report.py \
  --run-root "${PHASE6_RUN_ROOT:?PHASE6_RUN_ROOT is required}" \
  --report-root "${PHASE6_REPORT_ROOT:?PHASE6_REPORT_ROOT is required}"
