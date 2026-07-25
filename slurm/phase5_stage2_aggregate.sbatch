#!/usr/bin/env bash
#SBATCH --job-name=kam_p5_stage2_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${STAGE2_ENV:?STAGE2_ENV is required}/bin/activate"
cd "${STAGE2_REPO:?STAGE2_REPO is required}"
python -m kam.phase5.stage2_aggregate --run-root "${STAGE2_RUN_ROOT:?STAGE2_RUN_ROOT is required}" --report-root "${STAGE2_REPO:?STAGE2_REPO is required}/reports/phase5" --expected "${STAGE2_EXPECTED:?STAGE2_EXPECTED is required}" --stage-name "${STAGE2_NAME:?STAGE2_NAME is required}"
