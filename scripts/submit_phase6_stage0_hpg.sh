#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_repair_2541e09}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MANIFEST="${PHASE6_MANIFEST:-${HPG_REPO}/results/phase6/stage0/manifests/validity.jsonl}"
RUN_ROOT="${PHASE6_RUN_ROOT:-${HPG_REPO}/results/phase6/stage0/hpg_runs}"
LOG_ROOT="${PHASE6_LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase6/stage0}"
EXPECTED="${PHASE6_EXPECTED:-128}"

if [[ "${MODE}" == "--plan-only" ]]; then
  cat <<PLAN
Phase 6 Stage 0 HiPerGator plan
  repository: ${HPG_REPO}
  environment: ${HPG_ENV}
  account/qos/partition: ${HPG_ACCOUNT}/${HPG_QOS}/${HPG_PARTITION}
  manifest: ${MANIFEST}
  expected rows: ${EXPECTED}
  max concurrent jobs: ${MAX_CONCURRENT}
  run root: ${RUN_ROOT}
  logs: ${LOG_ROOT}
  scope: correctness and system microbenchmarks only; Stage 1+ remains gated
PLAN
  exit 0
fi

[[ -f "${MANIFEST}" ]] || { echo "Missing immutable manifest: ${MANIFEST}" >&2; exit 2; }
actual="$(wc -l < "${MANIFEST}")"
[[ "${actual}" -eq "${EXPECTED}" ]] || { echo "Manifest row count ${actual} != expected ${EXPECTED}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
exports="ALL,PHASE6_REPO=${HPG_REPO},PHASE6_ENV=${HPG_ENV},PHASE6_MANIFEST=${MANIFEST},PHASE6_RUN_ROOT=${RUN_ROOT},PHASE6_EXPECTED=${EXPECTED}"
array="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --export="${exports}" --array="0-$((EXPECTED - 1))%${MAX_CONCURRENT}" --output="${LOG_ROOT}/array_%A_%a.out" --error="${LOG_ROOT}/array_%A_%a.err" "${HPG_REPO}/slurm/phase6_stage0_array.sbatch")"
aggregate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --export="${exports}" --dependency="afterany:${array}" --output="${LOG_ROOT}/aggregate_%j.out" --error="${LOG_ROOT}/aggregate_%j.err" "${HPG_REPO}/slurm/phase6_stage0_aggregate.sbatch")"
echo "phase6_stage0_array=${array} phase6_stage0_aggregate=${aggregate}"
