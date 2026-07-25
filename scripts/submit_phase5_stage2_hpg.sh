#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---plan-only}"
STAGE2_ACCOUNT="${STAGE2_ACCOUNT:-uf-dsi}"
STAGE2_QOS="${STAGE2_QOS:-uf-dsi}"
STAGE2_PARTITION="${STAGE2_PARTITION:-hpg-turin}"
STAGE2_BLUE_ROOT="${STAGE2_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
STAGE2_REPO="${STAGE2_REPO:-${STAGE2_BLUE_ROOT}/KAM}"
STAGE2_ENV="${STAGE2_ENV:-${STAGE2_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${STAGE2_REPO}/results/phase5/stage2/manifests}"
RUN_ROOT="${RUN_ROOT:-${STAGE2_REPO}/results/phase5/stage2}"
LOG_ROOT="${LOG_ROOT:-${STAGE2_BLUE_ROOT}/logs/phase5/stage2}"

mkdir -p "${MANIFEST_ROOT}" "${RUN_ROOT}" "${LOG_ROOT}"
source "${STAGE2_ENV}/bin/activate"
if [[ ! -f "${MANIFEST_ROOT}/stage2A_component.jsonl" ]]; then
  python -m kam.phase5.stage2_manifest --output-dir "${MANIFEST_ROOT}"
fi

declare -A EXPECTED=( [stage2A_component]=450 [stage2B_capacity]=480 [stage2C_factorial]=600 [stage2D_symbolic]=60 )
declare -A LABEL=( [stage2A_component]=component [stage2B_capacity]=capacity [stage2C_factorial]=factorial [stage2D_symbolic]=symbolic )
if [[ "${MODE}" == "--plan-only" ]]; then
  echo "Phase V Stage 2 HPG plan"
  for stage in stage2A_component stage2B_capacity stage2C_factorial stage2D_symbolic; do
    echo "  ${stage}: $(wc -l < "${MANIFEST_ROOT}/${stage}.jsonl") rows; expected ${EXPECTED[$stage]}"
  done
  echo "  concurrency: ${MAX_CONCURRENT_GPUS}"
  echo "  reports: one afterok dependency per sub-stage"
  exit 0
fi

submit_stage() {
  local stage="${1}"
  local manifest="${MANIFEST_ROOT}/${stage}.jsonl"
  local root="${RUN_ROOT}/${stage}"
  local name="${LABEL[$stage]}"
  local expected="${EXPECTED[$stage]}"
  local exports="ALL,STAGE2_REPO=${STAGE2_REPO},STAGE2_ENV=${STAGE2_ENV},STAGE2_MANIFEST=${manifest},STAGE2_RUN_ROOT=${root},STAGE2_EXPECTED=${expected},STAGE2_NAME=${stage}"
  local array
  local report
  array="$(sbatch --parsable --account="${STAGE2_ACCOUNT}" --qos="${STAGE2_QOS}" --partition="${STAGE2_PARTITION}" --export="${exports}" --array="0-$((expected - 1))%${MAX_CONCURRENT_GPUS}" --output="${LOG_ROOT}/${stage}_%A_%a.out" --error="${LOG_ROOT}/${stage}_%A_%a.err" "${STAGE2_REPO}/slurm/phase5_stage2_array.sbatch")"
  report="$(sbatch --parsable --account="${STAGE2_ACCOUNT}" --qos="${STAGE2_QOS}" --partition="${STAGE2_PARTITION}" --export="${exports}" --dependency="afterok:${array}" --output="${LOG_ROOT}/${stage}_report_%j.out" --error="${LOG_ROOT}/${stage}_report_%j.err" "${STAGE2_REPO}/slurm/phase5_stage2_aggregate.sbatch")"
  echo "${stage}_array=${array} ${stage}_report=${report}"
}

submit_profile_stage() {
  local stage="${1}"
  local manifest="${MANIFEST_ROOT}/${stage}.jsonl"
  local root="${RUN_ROOT}/profile/${stage}"
  local expected="${EXPECTED[$stage]}"
  local exports="ALL,STAGE2_REPO=${STAGE2_REPO},STAGE2_ENV=${STAGE2_ENV},STAGE2_MANIFEST=${manifest},STAGE2_RUN_ROOT=${root},STAGE2_EXPECTED=3,STAGE2_NAME=${stage}_profile"
  local job
  job="$(sbatch --parsable --account="${STAGE2_ACCOUNT}" --qos="${STAGE2_QOS}" --partition="${STAGE2_PARTITION}" --export="${exports}" --array="0-2" --output="${LOG_ROOT}/${stage}_profile_%A_%a.out" --error="${LOG_ROOT}/${stage}_profile_%A_%a.err" "${STAGE2_REPO}/slurm/phase5_stage2_array.sbatch")"
  echo "${stage}_profile=${job} (rows 0-2; expected full stage size ${expected})"
}

if [[ "${MODE}" == "--profile" ]]; then
  submit_profile_stage stage2A_component
  submit_profile_stage stage2B_capacity
  submit_profile_stage stage2C_factorial
  submit_profile_stage stage2D_symbolic
  exit 0
fi

if [[ "${MODE}" == "--submit" ]]; then
  submit_stage stage2A_component
  submit_stage stage2B_capacity
  submit_stage stage2C_factorial
  submit_stage stage2D_symbolic
  exit 0
fi

echo "Usage: ${0} --plan-only|--profile|--submit" >&2
exit 2
