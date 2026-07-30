#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_atlas_s1_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
set -euo pipefail
source "${PHASE6_ATLAS_ENV:?PHASE6_ATLAS_ENV is required}/bin/activate"
cd "${PHASE6_ATLAS_REPO:?PHASE6_ATLAS_REPO is required}"
export PYTHONPATH="${PHASE6_ATLAS_REPO}:${PYTHONPATH:-}"
python -u -m kam.phase6.behavioral_atlas_stage1_analysis \
  --run-root "${PHASE6_ATLAS_STAGE1_ROOT:?PHASE6_ATLAS_STAGE1_ROOT is required}" \
  --report-root "${PHASE6_ATLAS_STAGE1_REPORT_ROOT:?PHASE6_ATLAS_STAGE1_REPORT_ROOT is required}" \
  --manifest "${PHASE6_ATLAS_STAGE1_MANIFEST:?PHASE6_ATLAS_STAGE1_MANIFEST is required}"
