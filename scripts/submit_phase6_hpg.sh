#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
STAGE="${PHASE6_STAGE:-stage1_mechanism}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_repair_2541e09}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
MANIFEST_MODE="${PHASE6_MANIFEST_MODE:-profile}"
CONFIG="${PHASE6_CONFIG:-${HPG_REPO}/configs/phase6/${STAGE}.yaml}"
MANIFEST="${PHASE6_MANIFEST:-${HPG_REPO}/results/phase6/${STAGE}/manifests/${MANIFEST_MODE}.jsonl}"
RUN_ROOT="${PHASE6_RUN_ROOT:-${HPG_REPO}/results/phase6/${STAGE}/hpg_runs}"
LOG_ROOT="${PHASE6_LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase6/${STAGE}}"
REPORT_ROOT="${PHASE6_REPORT_ROOT:-${HPG_REPO}/reports/phase6}"
GATE="${PHASE6_STAGE0_GATE:-${HPG_REPO}/reports/phase6/PHASE6_STAGE0_GATE_HPG.json}"

if [[ "${MANIFEST_MODE}" != "profile" && "${MANIFEST_MODE}" != "full" ]]; then
  echo "PHASE6_MANIFEST_MODE must be profile or full" >&2
  exit 2
fi

if [[ "${MODE}" == "--plan-only" ]]; then
  echo "Phase 6 HPG plan"
  echo "  stage/config: ${STAGE} / ${CONFIG}"
  echo "  manifest mode: ${MANIFEST_MODE}"
  echo "  repository/environment: ${HPG_REPO} / ${HPG_ENV}"
  echo "  manifest/run root: ${MANIFEST} / ${RUN_ROOT}"
  echo "  max concurrent: ${MAX_CONCURRENT}"
  echo "  resource classes must be profiled with sinfo before submission"
  if command -v sinfo >/dev/null 2>&1; then sinfo -o "%P %G %m %l" | sed -n '1,12p'; fi
  exit 0
fi

source "${HPG_ENV}/bin/activate"
mkdir -p "$(dirname "${MANIFEST}")" "${RUN_ROOT}" "${LOG_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then python -m kam.phase6.manifest --config "${CONFIG}" --mode "${MANIFEST_MODE}" --output "${MANIFEST}"; fi
count="$(wc -l < "${MANIFEST}")"
if [[ "${MODE}" == "--submit" && "${STAGE}" != "stage0_validity" ]]; then
  [[ -f "${GATE}" ]] || { echo "Missing upstream Stage 0 gate: ${GATE}" >&2; exit 2; }
  python - "${GATE}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("stage0_pass", False):
    raise SystemExit("Stage 0 gate is not passing; large stage remains blocked")
PY
fi
exports="ALL,PHASE6_STAGE=${STAGE},PHASE6_REPO=${HPG_REPO},PHASE6_ENV=${HPG_ENV},PHASE6_MANIFEST=${MANIFEST},PHASE6_RUN_ROOT=${RUN_ROOT},PHASE6_REPORT_ROOT=${REPORT_ROOT},PHASE6_EXPECTED=${count}"
array="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --export="${exports}" --array="0-$((count - 1))%${MAX_CONCURRENT}" --output="${LOG_ROOT}/array_%A_%a.out" --error="${LOG_ROOT}/array_%A_%a.err" "${HPG_REPO}/slurm/phase6_array.sbatch")"
aggregate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --export="${exports}" --dependency="afterany:${array}" --output="${LOG_ROOT}/aggregate_%j.out" --error="${LOG_ROOT}/aggregate_%j.err" "${HPG_REPO}/slurm/phase6_aggregate.sbatch")"
echo "stage=${STAGE} array=${array} aggregate=${aggregate} rows=${count}"
