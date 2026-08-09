#!/usr/bin/env bash
#SBATCH --job-name=kam_p6_router_replay_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PHASE6_ROUTER_FIX_ENV:?required}/bin/activate"
cd "${PHASE6_ROUTER_FIX_REPO:?required}"
export PYTHONPATH="${PHASE6_ROUTER_FIX_REPO}:${PYTHONPATH:-}"
python -u -m kam.phase6.behavioral_atlas_router_fix_campaign replay-aggregate \
  --manifest "${PHASE6_ROUTER_REPLAY_MANIFEST:?required}" \
  --output-root "${PHASE6_ROUTER_REPLAY_ROOT:?required}" \
  --report-root "${PHASE6_ROUTER_REPLAY_REPORT_ROOT:?required}"
