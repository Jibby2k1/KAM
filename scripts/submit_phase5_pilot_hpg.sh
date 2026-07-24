#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
PHASE5_CONFIG="${PHASE5_CONFIG:-${HPG_REPO}/configs/phase5/pilot.yaml}"
PHASE5_MANIFEST="${PHASE5_MANIFEST:-${HPG_REPO}/results/phase5/manifests/pilot.jsonl}"
PHASE5_RUN_ROOT="${PHASE5_RUN_ROOT:-${HPG_REPO}/results/phase5/pilot}"
LOG_ROOT="${LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase5}"

if [[ ! -f "${PHASE5_MANIFEST}" ]]; then
  source "${HPG_ENV}/bin/activate"
  python -m kam.phase5.pilot_manifest --config "${PHASE5_CONFIG}" --output "${PHASE5_MANIFEST}"
fi
ROW_COUNT="$(wc -l < "${PHASE5_MANIFEST}")"
EXPORTS="ALL,HPG_REPO=${HPG_REPO},HPG_ENV=${HPG_ENV},PHASE5_MANIFEST=${PHASE5_MANIFEST},PHASE5_RUN_ROOT=${PHASE5_RUN_ROOT}"
COMMON=(--parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --export="${EXPORTS}")

if [[ "${MODE}" == "--plan-only" ]]; then
  cat <<PLAN
Phase V Stage 1 mechanism-pilot HPG plan
  repository: ${HPG_REPO}
  environment: ${HPG_ENV}
  partition: ${HPG_PARTITION}
  rows: ${ROW_COUNT}
  target: 4 tasks x 6 variants x 2 active-capacity scales x 3 paired seeds = 144
  manifest: ${PHASE5_MANIFEST}
  run root: ${PHASE5_RUN_ROOT}
  concurrency: ${MAX_CONCURRENT_GPUS}
  protocol: iid_window_training
  report dependency: afterok array
PLAN
  exit 0
fi

if [[ "${MODE}" == "--profile" ]]; then
  mkdir -p "${LOG_ROOT}" "${PHASE5_RUN_ROOT}"
  PROFILE_INDICES="${PROFILE_INDICES:-0,1,2,18,19,20}"
  array_job="$(sbatch "${COMMON[@]}" --array="${PROFILE_INDICES}" --output="${LOG_ROOT}/pilot_profile_%A_%a.out" --error="${LOG_ROOT}/pilot_profile_%A_%a.err" "${HPG_REPO}/slurm/phase5_pilot_array.sbatch" | tail -1)"
  echo "profile_array=${array_job} indices=${PROFILE_INDICES}"
  exit 0
fi

if [[ "${MODE}" == "--submit" ]]; then
  mkdir -p "${LOG_ROOT}" "${PHASE5_RUN_ROOT}"
  array_job="$(sbatch "${COMMON[@]}" --array="0-$((ROW_COUNT - 1))%${MAX_CONCURRENT_GPUS}" --output="${LOG_ROOT}/pilot_%A_%a.out" --error="${LOG_ROOT}/pilot_%A_%a.err" "${HPG_REPO}/slurm/phase5_pilot_array.sbatch" | tail -1)"
  report_job="$(sbatch "${COMMON[@]}" --dependency="afterok:${array_job}" --output="${LOG_ROOT}/pilot_report_%j.out" --error="${LOG_ROOT}/pilot_report_%j.err" "${HPG_REPO}/slurm/phase5_pilot_aggregate.sbatch" | tail -1)"
  echo "array=${array_job} report=${report_job}"
  exit 0
fi

echo "Usage: ${0} --plan-only|--profile|--submit" >&2
exit 2
