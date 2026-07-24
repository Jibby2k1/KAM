#!/usr/bin/env bash
#SBATCH --job-name=kam_p3_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
: "${HPG_REPO:?HPG_REPO is required}"
: "${HPG_ENV:?HPG_ENV is required}"
source "$HPG_ENV/bin/activate"
cd "$HPG_REPO"
GATE_TYPE="${PHASE3_GATE_TYPE:-primary}"
GATE_OUTPUT="${PHASE3_GATE_OUTPUT:-${PHASE3_RUN_ROOT:?PHASE3_RUN_ROOT is required}/gates/phase3_gate.json}"
mkdir -p "$(dirname "$GATE_OUTPUT")"
if [[ "$GATE_TYPE" == "audit" ]]; then
  python -m kam.phase3.gate \
    --gate-type audit \
    --root "$HPG_REPO" \
    --output "$GATE_OUTPUT"
else
  python -m kam.phase3.gate \
    --gate-type primary \
    --aggregate "${PHASE3_RUN_ROOT:?PHASE3_RUN_ROOT is required}/phase3_aggregate.json" \
    --gate-config "$HPG_REPO/configs/phase3/gates.yaml" \
    --output "$GATE_OUTPUT"
fi
