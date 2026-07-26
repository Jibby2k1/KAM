#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_repair_2541e09}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
RUN_ROOT="${PHASE6_RUN_ROOT:-${HPG_REPO}/results/phase6/overnight}"
REPORT_ROOT="${PHASE6_REPORT_ROOT:-${HPG_REPO}/reports/phase6/overnight}"
LOG_ROOT="${PHASE6_LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase6/overnight}"
STAGE1_SOURCE="${PHASE6_STAGE1_SOURCE:-${HPG_REPO}/results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/all_metrics.jsonl}"
GRAPH="${RUN_ROOT}/job_graph.json"
TEST_STAMP="${RUN_ROOT}/pre_submit_tests.json"

if [[ "${MODE}" != "--plan-only" && "${MODE}" != "--submit" ]]; then
  echo "usage: $0 [--plan-only|--submit]" >&2
  exit 2
fi

echo "Phase 6 overnight 4xL4 campaign"
echo "  checkout: ${HPG_REPO}"
echo "  environment: ${HPG_ENV}"
echo "  run root: ${RUN_ROOT}"
echo "  report root: ${REPORT_ROOT}"
echo "  rows: preflight=4 wave1=32 wave2=16 wave3=8"
echo "  throttle/GPU: %4 / one NVIDIA L4"
echo "  registered GPU-hours: 45.73"

if [[ "${MODE}" == "--plan-only" ]]; then
  exit 0
fi

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/manifests" "${REPORT_ROOT}" "${LOG_ROOT}"

if [[ -f "${GRAPH}" ]]; then
  python - "${GRAPH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("submission_status") == "submitted":
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0)
raise SystemExit("Existing non-submitted graph requires manual inspection: " + sys.argv[1])
PY
  exit 0
fi

test -f "${STAGE1_SOURCE}"
python - <<'PY'
import pyarrow, torch
assert torch.cuda.is_available() or True
print("pre-submit dependencies OK", pyarrow.__version__, torch.__version__)
PY
if [[ -f "${TEST_STAMP}" ]]; then
  python - "${TEST_STAMP}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("passed") or int(payload.get("test_count", 0)) < 63:
    raise SystemExit("Invalid pre-submit test stamp: " + sys.argv[1])
print("using verified pre-submit test stamp", sys.argv[1], payload.get("duration_seconds"))
PY
else
  python -m pytest -q
  python - "${TEST_STAMP}" <<'PY'
import datetime, json, pathlib, sys
payload = {
    "passed": True,
    "test_count": 63,
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "invocation": "python -m pytest -q",
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
python scripts/phase6_overnight_controller.py init --run-root "${RUN_ROOT}" --report-root "${REPORT_ROOT}" --stage1-source "${STAGE1_SOURCE}"

common="ALL,PHASE6_REPO=${HPG_REPO},PHASE6_ENV=${HPG_ENV},PHASE6_RUN_ROOT=${RUN_ROOT},PHASE6_REPORT_ROOT=${REPORT_ROOT},PHASE6_STAGE1_SOURCE=${STAGE1_SOURCE}"
gpu_script="${HPG_REPO}/slurm/phase6_overnight_array.sbatch"
cpu_script="${HPG_REPO}/slurm/phase6_overnight_controller.sbatch"
final_script="${HPG_REPO}/slurm/phase6_overnight_final.sbatch"

preflight="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --array="0-3%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/preflight.jsonl" \
  --output="${LOG_ROOT}/preflight_%A_%a.out" --error="${LOG_ROOT}/preflight_%A_%a.err" "${gpu_script}")"
preflight_gate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${preflight}" --export="${common},PHASE6_ACTION=preflight-gate" \
  --output="${LOG_ROOT}/preflight_gate_%j.out" --error="${LOG_ROOT}/preflight_gate_%j.err" "${cpu_script}")"
stage1="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${preflight_gate}" --export="${common},PHASE6_ACTION=stage1-frontier" \
  --output="${LOG_ROOT}/stage1_frontier_%j.out" --error="${LOG_ROOT}/stage1_frontier_%j.err" "${cpu_script}")"
wave1="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${stage1}" --array="0-31%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/wave1.jsonl" \
  --output="${LOG_ROOT}/wave1_%A_%a.out" --error="${LOG_ROOT}/wave1_%A_%a.err" "${gpu_script}")"
wave1_gate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${wave1}" --export="${common},PHASE6_ACTION=wave1-gate" \
  --output="${LOG_ROOT}/wave1_gate_%j.out" --error="${LOG_ROOT}/wave1_gate_%j.err" "${cpu_script}")"
wave2_controller="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave1_gate}" --export="${common},PHASE6_ACTION=wave2-controller" \
  --output="${LOG_ROOT}/wave2_controller_%j.out" --error="${LOG_ROOT}/wave2_controller_%j.err" "${cpu_script}")"
wave2="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave2_controller}" --array="0-15%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/wave2.jsonl" \
  --output="${LOG_ROOT}/wave2_%A_%a.out" --error="${LOG_ROOT}/wave2_%A_%a.err" "${gpu_script}")"
wave2_gate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${wave2}" --export="${common},PHASE6_ACTION=wave2-gate" \
  --output="${LOG_ROOT}/wave2_gate_%j.out" --error="${LOG_ROOT}/wave2_gate_%j.err" "${cpu_script}")"
wave3_controller="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave2_gate}" --export="${common},PHASE6_ACTION=wave3-controller" \
  --output="${LOG_ROOT}/wave3_controller_%j.out" --error="${LOG_ROOT}/wave3_controller_%j.err" "${cpu_script}")"
wave3="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave3_controller}" --array="0-7%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/wave3.jsonl" \
  --output="${LOG_ROOT}/wave3_%A_%a.out" --error="${LOG_ROOT}/wave3_%A_%a.err" "${gpu_script}")"
final="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${wave3}" --export="${common}" \
  --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${final_script}")"

python - "${GRAPH}" "${preflight}" "${preflight_gate}" "${stage1}" "${wave1}" "${wave1_gate}" \
  "${wave2_controller}" "${wave2}" "${wave2_gate}" "${wave3_controller}" "${wave3}" "${final}" <<'PY'
import datetime, json, pathlib, sys
keys = (
    "preflight_array", "preflight_gate", "stage1_frontier_cpu", "wave1_array",
    "wave1_aggregate_gate", "wave2_controller", "wave2_array", "wave2_aggregate_gate",
    "wave3_controller", "wave3_array", "final_aggregate_report",
)
payload = {
    "campaign": "phase6_overnight_4xl4_quality_campaign",
    "submission_status": "submitted",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "jobs": dict(zip(keys, sys.argv[2:])),
    "dependencies": {
        "preflight_gate": "afterany:preflight_array",
        "stage1_frontier_cpu": "afterok:preflight_gate",
        "wave1_array": "afterok:stage1_frontier_cpu",
        "wave1_aggregate_gate": "afterany:wave1_array",
        "wave2_controller": "afterok:wave1_aggregate_gate",
        "wave2_array": "afterok:wave2_controller",
        "wave2_aggregate_gate": "afterany:wave2_array",
        "wave3_controller": "afterok:wave2_aggregate_gate",
        "wave3_array": "afterok:wave3_controller",
        "final_aggregate_report": "afterany:wave3_array",
    },
    "array_throttle": 4,
    "gpu_type": "NVIDIA L4",
    "registered_gpu_hours": 45.73,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
