#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${PHASE6_ENV:?PHASE6_ENV is required}/bin/activate"
cd "${PHASE6_REPO:?PHASE6_REPO is required}"
python -u -m kam.phase6.aggregate --generic --run-root "${PHASE6_RUN_ROOT:?PHASE6_RUN_ROOT is required}" --expected "${PHASE6_EXPECTED:?PHASE6_EXPECTED is required}" --stage "${PHASE6_STAGE:?PHASE6_STAGE is required}" --report-root "${PHASE6_REPORT_ROOT:-${PHASE6_REPO}/reports/phase6}"
