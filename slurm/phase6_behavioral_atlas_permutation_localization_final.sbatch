#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_perm_loc_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PHASE6_LOCALIZATION_ENV:?PHASE6_LOCALIZATION_ENV is required}/bin/activate"
cd "${PHASE6_LOCALIZATION_REPO:?PHASE6_LOCALIZATION_REPO is required}"
export PYTHONPATH="${PHASE6_LOCALIZATION_REPO}:${PYTHONPATH:-}"
python -u -m kam.phase6.behavioral_atlas_permutation_localization aggregate \
  --manifest "${PHASE6_LOCALIZATION_MANIFEST:?PHASE6_LOCALIZATION_MANIFEST is required}" \
  --output-root "${PHASE6_LOCALIZATION_ROOT:?PHASE6_LOCALIZATION_ROOT is required}" \
  --report-root "${PHASE6_LOCALIZATION_REPORT_ROOT:?PHASE6_LOCALIZATION_REPORT_ROOT is required}"
