#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_perm_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PHASE6_ATLAS_ENV:?PHASE6_ATLAS_ENV is required}/bin/activate"
cd "${PHASE6_ATLAS_REPO:?PHASE6_ATLAS_REPO is required}"
export PYTHONPATH="${PHASE6_ATLAS_REPO}:${PYTHONPATH:-}"
python -u -m kam.phase6.behavioral_atlas_permutation_audit aggregate \
  --manifest "${PHASE6_ATLAS_PERMUTATION_AUDIT_MANIFEST:?PHASE6_ATLAS_PERMUTATION_AUDIT_MANIFEST is required}" \
  --output-root "${PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT:?PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT is required}" \
  --report-root "${PHASE6_ATLAS_PERMUTATION_AUDIT_REPORT_ROOT:?PHASE6_ATLAS_PERMUTATION_AUDIT_REPORT_ROOT is required}"
