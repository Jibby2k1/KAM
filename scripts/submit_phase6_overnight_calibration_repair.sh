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
LOG_ROOT="${PHASE6_LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase6/overnight_calibration_repair}"
STAGE1_SOURCE="${PHASE6_STAGE1_SOURCE:-${HPG_REPO}/results/phase6/stage1_mechanism/hpg_runs_full_taskfix3/all_metrics.jsonl}"
PRIOR_GRAPH="${RUN_ROOT}/timeout_repair_job_graph.json"
REPAIR_GRAPH="${RUN_ROOT}/calibration_fallback_repair_job_graph.json"
REPAIR_MANIFEST="${RUN_ROOT}/manifests/wave1_calibration_fallback_repair.jsonl"

if [[ "${MODE}" != "--plan-only" && "${MODE}" != "--submit" ]]; then
  echo "usage: $0 [--plan-only|--submit]" >&2
  exit 2
fi

echo "Phase 6 architecture-calibration fallback repair"
echo "  preserve: every completed Wave 1 row"
echo "  rerun: only rows still missing after canceling the superseded repair"
echo "  Wave 1 limit: 8 hours; Wave 2/3 limits: 24 hours"
echo "  run root: ${RUN_ROOT}"
if [[ "${MODE}" == "--plan-only" ]]; then
  exit 0
fi

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}"
test -f "${PRIOR_GRAPH}"

if [[ -f "${REPAIR_GRAPH}" ]]; then
  python - "${REPAIR_GRAPH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("submission_status") != "submitted":
    raise SystemExit("Existing calibration repair graph is not a completed submission record")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  exit 0
fi

# Stop the exact superseded graph first. Waiting for every active array task to
# leave the queue prevents an old row ID from being written after revision 2.
mapfile -t prior_jobs < <(
  python - "${PRIOR_GRAPH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for value in payload["jobs"].values():
    print(value)
PY
)
if (( ${#prior_jobs[@]} )); then
  # Completed IDs may cause harmless scheduler diagnostics.
  scancel "${prior_jobs[@]}" 2>/dev/null || true
fi
prior_array="$(
  python - "${PRIOR_GRAPH}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["jobs"]["wave1_timeout_repair_array"])
PY
)"
for _ in $(seq 1 60); do
  if [[ -z "$(squeue -h -j "${prior_array}")" ]]; then
    break
  fi
  sleep 1
done
if [[ -n "$(squeue -h -j "${prior_array}")" ]]; then
  echo "Superseded Wave 1 repair still has active tasks after cancellation" >&2
  exit 1
fi

python scripts/phase6_overnight_controller.py repair-wave1-calibration \
  --run-root "${RUN_ROOT}" --report-root "${REPORT_ROOT}" --stage1-source "${STAGE1_SOURCE}"
repair_count="$(python - "${REPAIR_MANIFEST}" <<'PY'
import pathlib, sys
print(sum(bool(line.strip()) for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))
PY
)"
python -m kam.phase6.overnight_analysis validate-manifest \
  --path "${REPAIR_MANIFEST}" --expected "${repair_count}"
repair_last="$((repair_count - 1))"

common="ALL,PHASE6_REPO=${HPG_REPO},PHASE6_ENV=${HPG_ENV},PHASE6_RUN_ROOT=${RUN_ROOT},PHASE6_REPORT_ROOT=${REPORT_ROOT},PHASE6_STAGE1_SOURCE=${STAGE1_SOURCE}"
gpu_script="${HPG_REPO}/slurm/phase6_overnight_array.sbatch"
cpu_script="${HPG_REPO}/slurm/phase6_overnight_controller.sbatch"
final_script="${HPG_REPO}/slurm/phase6_overnight_final.sbatch"

repair="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --time="08:00:00" --array="0-${repair_last}%4" --export="${common},PHASE6_MANIFEST=${REPAIR_MANIFEST}" \
  --output="${LOG_ROOT}/wave1_%A_%a.out" --error="${LOG_ROOT}/wave1_%A_%a.err" "${gpu_script}")"
wave1_gate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${repair}" --export="${common},PHASE6_ACTION=wave1-gate" \
  --output="${LOG_ROOT}/wave1_gate_%j.out" --error="${LOG_ROOT}/wave1_gate_%j.err" "${cpu_script}")"
wave2_controller="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave1_gate}" --export="${common},PHASE6_ACTION=wave2-controller" \
  --output="${LOG_ROOT}/wave2_controller_%j.out" --error="${LOG_ROOT}/wave2_controller_%j.err" "${cpu_script}")"
wave2="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --time="24:00:00" --dependency="afterok:${wave2_controller}" --array="0-15%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/wave2.jsonl" \
  --output="${LOG_ROOT}/wave2_%A_%a.out" --error="${LOG_ROOT}/wave2_%A_%a.err" "${gpu_script}")"
wave2_gate="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${wave2}" --export="${common},PHASE6_ACTION=wave2-gate" \
  --output="${LOG_ROOT}/wave2_gate_%j.out" --error="${LOG_ROOT}/wave2_gate_%j.err" "${cpu_script}")"
wave3_controller="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterok:${wave2_gate}" --export="${common},PHASE6_ACTION=wave3-controller" \
  --output="${LOG_ROOT}/wave3_controller_%j.out" --error="${LOG_ROOT}/wave3_controller_%j.err" "${cpu_script}")"
wave3="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --time="24:00:00" --dependency="afterok:${wave3_controller}" --array="0-7%4" --export="${common},PHASE6_MANIFEST=${RUN_ROOT}/manifests/wave3.jsonl" \
  --output="${LOG_ROOT}/wave3_%A_%a.out" --error="${LOG_ROOT}/wave3_%A_%a.err" "${gpu_script}")"
final="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" \
  --dependency="afterany:${wave3}" --export="${common}" \
  --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${final_script}")"

python - "${REPAIR_GRAPH}" "${repair_count}" "${repair}" "${wave1_gate}" "${wave2_controller}" "${wave2}" \
  "${wave2_gate}" "${wave3_controller}" "${wave3}" "${final}" <<'PY'
import datetime, json, pathlib, sys
keys = (
    "wave1_calibration_repair_array", "wave1_repair_gate", "wave2_controller",
    "wave2_array", "wave2_aggregate_gate", "wave3_controller", "wave3_array",
    "final_aggregate_report",
)
payload = {
    "campaign": "phase6_overnight_4xl4_calibration_fallback_repair",
    "submission_status": "submitted",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "jobs": dict(zip(keys, sys.argv[3:])),
    "preserved_wave1_rows": 32 - int(sys.argv[2]),
    "repaired_wave1_rows": int(sys.argv[2]),
    "array_throttle": 4,
    "gpu_type": "NVIDIA L4",
    "budget_policy": "architecture_specific_calibration_or_registered_minimum",
    "wall_limits": {"wave1_repair": "08:00:00", "wave2": "24:00:00", "wave3": "24:00:00"},
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
