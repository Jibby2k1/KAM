#!/usr/bin/env bash
#SBATCH --job-name=kam_p5_pilot_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${HPG_ENV:?HPG_ENV is required}/bin/activate"
cd "${HPG_REPO:?HPG_REPO is required}"
python -m kam.phase5.pilot_aggregate --run-root "${PHASE5_RUN_ROOT:?PHASE5_RUN_ROOT is required}" --report-root "${HPG_REPO:?HPG_REPO is required}/reports/phase5" --expected 144
