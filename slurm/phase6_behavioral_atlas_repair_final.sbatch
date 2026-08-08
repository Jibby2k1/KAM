#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_atlas_repair_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail
source "${PHASE6_ATLAS_ENV:?PHASE6_ATLAS_ENV is required}/bin/activate"
cd "${PHASE6_ATLAS_REPO:?PHASE6_ATLAS_REPO is required}"
export PYTHONPATH="${PHASE6_ATLAS_REPO}:${PYTHONPATH:-}"
python -u -m kam.phase6.behavioral_atlas_repair audit \
  --stage0-run-root "${PHASE6_ATLAS_STAGE0_ROOT:?PHASE6_ATLAS_STAGE0_ROOT is required}" \
  --stage0-manifest "${PHASE6_ATLAS_STAGE0_MANIFEST:?PHASE6_ATLAS_STAGE0_MANIFEST is required}" \
  --repair-root "${PHASE6_ATLAS_REPAIR_ROOT:?PHASE6_ATLAS_REPAIR_ROOT is required}" \
  --report-root "${PHASE6_ATLAS_REPAIR_REPORT_ROOT:?PHASE6_ATLAS_REPAIR_REPORT_ROOT is required}"
