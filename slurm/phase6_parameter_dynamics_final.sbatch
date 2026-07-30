#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_pd_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail
source "${PHASE6_PD_ENV:?PHASE6_PD_ENV is required}/bin/activate"
cd "${PHASE6_PD_REPO:?PHASE6_PD_REPO is required}"
export PYTHONPATH="${PHASE6_PD_REPO}:${PYTHONPATH:-}"
python -u scripts/build_phase6_parameter_dynamics_report.py \
  --run-root "${PHASE6_PD_RUN_ROOT:?PHASE6_PD_RUN_ROOT is required}" \
  --report-root "${PHASE6_PD_REPORT_ROOT:?PHASE6_PD_REPORT_ROOT is required}" \
  --manifest "${PHASE6_PD_MANIFEST:?PHASE6_PD_MANIFEST is required}"
