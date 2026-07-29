#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_confirmation_v2_20260728}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
RUN_ROOT="${PHASE6_RUN_ROOT:-${HPG_BLUE_ROOT}/KAM_confirmation_v2_results/results/phase6/confirmation_v2}"
REPORT_ROOT="${PHASE6_REPORT_ROOT:-${HPG_BLUE_ROOT}/KAM_confirmation_v2_results/reports/phase6/confirmation_v2}"
LOG_ROOT="${PHASE6_LOG_ROOT:-${HPG_BLUE_ROOT}/KAM_confirmation_v2_results/logs/phase6/confirmation_v2}"
MANIFEST="${RUN_ROOT}/manifest.jsonl"
GRAPH="${RUN_ROOT}/job_graph.json"
CORPUS_ROOT="${HPG_REPO}/data/phase6_confirmation"

if [[ "${MODE}" != "--plan-only" && "${MODE}" != "--submit" ]]; then
  echo "usage: $0 [--plan-only|--submit]" >&2
  exit 2
fi

echo "Phase 6 confirmation v2"
echo "  checkout: ${HPG_REPO}"
echo "  run root: ${RUN_ROOT}"
echo "  report root: ${REPORT_ROOT}"
echo "  rows: 156 (30 primary pairs, 24 replication pairs, 12 secondary-control pairs, 24 mechanism rows)"
echo "  throttle/GPU: %4 / one NVIDIA L4"
echo "  fixed sample: no optional stopping"

if [[ "${MODE}" == "--plan-only" ]]; then
  exit 0
fi

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/rows/confirmation_v2" "${REPORT_ROOT}" "${LOG_ROOT}"

if [[ -f "${GRAPH}" ]]; then
  python - "${GRAPH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("submission_status") == "submitted":
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0)
raise SystemExit("Existing non-submitted graph requires inspection: " + sys.argv[1])
PY
  exit 0
fi

test -s "${CORPUS_ROOT}/corpus_manifest.json"
test -s "${CORPUS_ROOT}/TinyStoriesV2-GPT4-train.128MiB.txt"
test -s "${CORPUS_ROOT}/TinyStories-valid.validation.txt"
test -s "${CORPUS_ROOT}/TinyStories-valid.test.txt"
test -s "${HPG_REPO}/data/tinyshakespeare.txt"

python scripts/build_phase6_confirmation_manifest.py --output "${MANIFEST}"
python -m pytest -q tests/test_phase6_confirmation.py tests/test_phase6_components.py

common="ALL,PHASE6_REPO=${HPG_REPO},PHASE6_ENV=${HPG_ENV},PHASE6_RUN_ROOT=${RUN_ROOT},PHASE6_REPORT_ROOT=${REPORT_ROOT},PHASE6_MANIFEST=${MANIFEST}"
array_script="${HPG_REPO}/slurm/phase6_overnight_array.sbatch"
final_script="${HPG_REPO}/slurm/phase6_confirmation_final.sbatch"

array_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --time="04:00:00" --array="0-155%4" --export="${common}" \
  --output="${LOG_ROOT}/row_%A_%a.out" --error="${LOG_ROOT}/row_%A_%a.err" "${array_script}")"
final_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${array_job}" --export="${common}" \
  --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${final_script}")"

python - "${GRAPH}" "${MANIFEST}" "${array_job}" "${final_job}" <<'PY'
import datetime, hashlib, json, pathlib, sys
graph, manifest, array_job, final_job = sys.argv[1:]
payload_bytes = pathlib.Path(manifest).read_bytes()
payload = {
    "campaign": "phase6_confirmation_v2",
    "submission_status": "submitted",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest": manifest,
    "manifest_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    "row_count": 156,
    "fixed_sample_no_optional_stopping": True,
    "jobs": {"confirmation_array": array_job, "final_report": final_job},
    "dependencies": {"final_report": "afterany:confirmation_array"},
    "array_throttle": 4,
    "gpu_type": "NVIDIA L4",
}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
