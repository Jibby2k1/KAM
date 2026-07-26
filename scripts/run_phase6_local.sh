#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---profile}"
STAGE="${PHASE6_STAGE:-stage1_mechanism}"
CONFIG="${PHASE6_CONFIG:-configs/phase6/${STAGE}.yaml}"
MANIFEST="${PHASE6_MANIFEST:-results/phase6/${STAGE}/manifests/profile.jsonl}"
RUN_ROOT="${PHASE6_RUN_ROOT:-results/phase6/${STAGE}/local_runs}"
DEVICE="${PHASE6_DEVICE:-auto}"

if [[ "${MODE}" == "--plan-only" ]]; then
  echo "Phase 6 local plan: stage=${STAGE} config=${CONFIG} manifest=${MANIFEST} run_root=${RUN_ROOT} device=${DEVICE}"
  exit 0
fi

python -m kam.phase6.manifest --config "${CONFIG}" --mode "${MODE#--}" --output "${MANIFEST}"
mkdir -p "${RUN_ROOT}"
count="$(wc -l < "${MANIFEST}")"
for ((index = 0; index < count; index++)); do
  python -m kam.phase6.run_array --manifest "${MANIFEST}" --array-index "${index}" --output "${RUN_ROOT}/row_${index}.json" --run-root "${RUN_ROOT}" --device "${DEVICE}" --resume
done
python -m kam.phase6.aggregate --generic --run-root "${RUN_ROOT}" --expected "${count}" --stage "${STAGE}" --report-root "reports/phase6"
