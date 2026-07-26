#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_s0_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
source "${PHASE6_ENV:?PHASE6_ENV is required}/bin/activate"
cd "${PHASE6_REPO:?PHASE6_REPO is required}"
python -u -m kam.phase6.aggregate --run-root "${PHASE6_RUN_ROOT:?PHASE6_RUN_ROOT is required}" --expected "${PHASE6_EXPECTED:?PHASE6_EXPECTED is required}" --report-root "${PHASE6_REPO:?PHASE6_REPO is required}/reports/phase6"
